import pacai.core.action
import pacai.core.agentinfo
from pacai.agents.greedy import GreedyFeatureAgent
from pacai.core.board import Position
from pacai.search.distance import DistancePreComputer


def create_team() -> list[pacai.core.agentinfo.AgentInfo]:
    """
    Get the agent information that will be used to create a capture team.
    """
    agent1 = pacai.core.agentinfo.AgentInfo(name=f"{__name__}.OffensiveAgent")
    agent2 = pacai.core.agentinfo.AgentInfo(name=f"{__name__}.DefensiveAgent")
    return [agent1, agent2]


class OffensiveAgent(GreedyFeatureAgent):
    """Offensive agent: prioritize eating food while avoiding enemy Pac-Men."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._distances = DistancePreComputer()
        self.weights = {
            "successorScore": 100,
            "distanceToFood": -60,
            "enemyDistance": 20,
            "enemyTooClose": -250,
            "stopped": -100,
            "reverse": -8,
            "carryHome": -15,
        }

    def game_start(self, initial_state):
        super().game_start(initial_state)
        # Precompute distances for the entire board.
        self._distances.compute(initial_state.board)

        # Remember how much food the enemy started with.
        self._start_food = initial_state.food_count(agent_index=self.agent_index)

        # Find walkable border positions on our side so we know where "home" is.
        board = initial_state.board
        mid = board.width // 2
        my_pos = initial_state.get_agent_position(self.agent_index)
        if my_pos.col < mid:
            border_col = mid - 1
        else:
            border_col = mid

        self._border_positions = [
            Position(row, border_col) for row in range(board.height)
            if not board.is_wall(Position(row, border_col))
        ]

    def compute_features(self, state, action):
        features = {}
        # `state` is already the successor for `action`.
        successor = state
        my_index = self.agent_index
        my_pos = successor.get_agent_position(my_index)

        if my_pos is None:
            return features

        # 1. Score of successor state (team-normalized).
        features["successorScore"] = successor.get_normalized_score(my_index)

        # 2. Distance to nearest food we can eat.
        food_positions = successor.get_food(agent_index=my_index)
        if food_positions:
            distances = [
                self._distances.get_distance(my_pos, food_pos)
                for food_pos in food_positions
            ]
            distances = [d for d in distances if d is not None]
            features["distanceToFood"] = min(distances) if distances else 0
        else:
            features["distanceToFood"] = 0

        # 3. Distance to non-scared opponents.
        opponents = successor.get_nonscared_opponent_positions(agent_index=my_index)
        opponent_positions = [
            pos
            for pos in opponents.values()
            if pos is not None
        ]

        enemy_distances = [
            self._distances.get_distance(my_pos, opp)
            for opp in opponent_positions
            if opp is not None
        ]
        enemy_distances = [d for d in enemy_distances if d is not None]

        # Check if enemy ghosts are scared (we ate a power capsule).
        scared_opponents = successor.get_scared_opponent_positions(
            agent_index=my_index
        )
        enemies_all_scared = (
            len(scared_opponents) > 0 and len(opponent_positions) == 0
        )

        if enemy_distances:
            min_enemy_distance = min(enemy_distances)
            features["enemyDistance"] = min_enemy_distance
            features["enemyTooClose"] = 1 if min_enemy_distance <= 3 else 0
        else:
            min_enemy_distance = None
            features["enemyDistance"] = 20
            features["enemyTooClose"] = 0

        # 4. Carry-and-return: pull toward home when carrying food on enemy side.
        if successor.is_pacman(my_index):
            current_food = successor.food_count(agent_index=my_index)
            carrying = self._start_food - current_food

            home_distances = [
                self._distances.get_distance(my_pos, bp)
                for bp in self._border_positions
            ]
            home_distances = [d for d in home_distances if d is not None]
            nearest_home = min(home_distances) if home_distances else 0

            if enemies_all_scared:
                # Ghosts are scared, keep eating safely.
                features["carryHome"] = 0
            elif carrying >= 3:
                features["carryHome"] = nearest_home
            else:
                features["carryHome"] = 0
        else:
            features["carryHome"] = 0

        # Avoid stopping and reversing.
        features["stopped"] = 1 if action == pacai.core.action.STOP else 0

        agent_actions = state.get_agent_actions(my_index)
        if len(agent_actions) > 1:
            # Reverse relative to the action from two moves ago (successor offset).
            previous_action = agent_actions[-2]
            reverse_action = state.get_reverse_action(previous_action)
            features["reverse"] = 1 if action == reverse_action else 0
        else:
            features["reverse"] = 0

        return features


class DefensiveAgent(GreedyFeatureAgent):
    """Defensive agent: prioritize stopping invaders on your side."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._distances = DistancePreComputer()

        self.weights = {
            "onDefense": 100,
            "invaderDistance": -100,
            "stopped": -100,
            "reverse": -8,
            "numInvaders": -1000,
            "invaderFoodDistance": -5,
            "urgentDefense": 200,
            "score": 50,
            "scaredFlee": -150,
            "patrolDistance": -8,
        }

    def game_start(self, initial_state):
        super().game_start(initial_state)
        # Precompute distances for the entire board.
        self._distances.compute(initial_state.board)

        # Find walkable border positions so we can patrol when idle.
        board = initial_state.board
        mid = board.width // 2
        my_pos = initial_state.get_agent_position(self.agent_index)
        if my_pos.col < mid:
            border_col = mid - 1
        else:
            border_col = mid

        self._border_positions = [
            Position(row, border_col) for row in range(board.height)
            if not board.is_wall(Position(row, border_col))
        ]

    def compute_features(self, state, action):
        features = {}
        # `state` is already the successor for `action`.
        successor = state
        my_index = self.agent_index
        my_pos = successor.get_agent_position(my_index)

        if my_pos is None:
            return features

        # 1. On defense? True if agent is not a Pac-Man (i.e., on our side).
        features["onDefense"] = 1 if not successor.is_pacman(my_index) else 0

        # 2. Find invaders on our side using the capture helper.
        invader_positions = successor.get_invader_positions(agent_index=my_index)
        
        features["numInvaders"] = len(invader_positions)

        # 3. Distance to nearest invader.
        scared = successor.is_scared(my_index)

        if invader_positions:
            distances = [
                self._distances.get_distance(my_pos, inv_pos)
                for inv_pos in invader_positions.values()
            ]
            distances = [d for d in distances if d is not None]
            closest = min(distances) if distances else 0

            if scared:
                # When scared, don't chase invaders
                features["invaderDistance"] = 0
                features["urgentDefense"] = 0
                features["invaderFoodDistance"] = 0
                # Flee if an invader is within 4 tiles.
                if closest <= 4:
                    features["scaredFlee"] = max(0, 5 - closest)
                else:
                    features["scaredFlee"] = 0
            else:
                features["invaderDistance"] = closest
                features["scaredFlee"] = 0

                # Make defense more urgent when invaders are nearby.
                features["urgentDefense"] = 1 / (closest + 1)

                invader_indices = list(invader_positions.keys())
                first_invader_index = invader_indices[0]
                our_food_positions = successor.get_food(
                    agent_index=first_invader_index
                )

                if our_food_positions:
                    distances_to_food = []
                    for inv_pos in invader_positions.values():
                        for food_pos in our_food_positions:
                            d = self._distances.get_distance(inv_pos, food_pos)
                            if d is not None:
                                distances_to_food.append(d)

                    features["invaderFoodDistance"] = (
                        min(distances_to_food) if distances_to_food else 0
                    )
                else:
                    features["invaderFoodDistance"] = 0
        else:
            features["invaderDistance"] = 0
            features["invaderFoodDistance"] = 0
            features["urgentDefense"] = 0
            features["scaredFlee"] = 0

            # 4. Patrol: when no invaders are visible, drift toward the border
            #    so we're ready to intercept.
            if self._border_positions:
                home_distances = [
                    self._distances.get_distance(my_pos, bp)
                    for bp in self._border_positions
                ]
                home_distances = [d for d in home_distances if d is not None]
                features["patrolDistance"] = (
                    min(home_distances) if home_distances else 0
                )
            else:
                features["patrolDistance"] = 0

        # Avoid stopping and reversing.
        features["stopped"] = 1 if action == pacai.core.action.STOP else 0

        agent_actions = state.get_agent_actions(my_index)

        if len(agent_actions) > 1:
            previous_action = agent_actions[-2]
            reverse_action = state.get_reverse_action(previous_action)
            features["reverse"] = 1 if action == reverse_action else 0
        else:
            features["reverse"] = 0
        
        features["score"] = state.get_normalized_score(my_index)

        return features
