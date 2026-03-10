import pacai.core.action
import pacai.core.agentinfo
from pacai.agents.greedy import GreedyFeatureAgent
from pacai.core.board import Position
from pacai.search.distance import DistancePreComputer


def create_team() -> list[pacai.core.agentinfo.AgentInfo]:
    """
    Get the agent information that will be used to create a capture team.

    Both members use the flex agent; role is decided each turn.
    """
    agent1 = pacai.core.agentinfo.AgentInfo(name=f"{__name__}.FlexAgent")
    agent2 = pacai.core.agentinfo.AgentInfo(name=f"{__name__}.FlexAgent")
    return [agent1, agent2]


# legacy classes removed; single FlexAgent below handles both roles


class FlexAgent(GreedyFeatureAgent):
    """A flexible capture agent that chooses offense/defense each turn."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._distances = DistancePreComputer()

        # weights for different roles
        self.offensive_weights = {
            "successorScore": 100,
            "distanceToFood": -60,
            "enemyDistance": -20,
            "enemyTooClose": -60,
            "stopped": -100,
            "reverse": -2,
            "carryHome": -5,
            "distanceToCapsule": -50,
        }

        self.defensive_weights = {
            "onDefense": 100,
            "invaderDistance": -60,
            "stopped": -100,
            "reverse": -2,
            "numInvaders": -1000,
            "invaderFoodDistance": -10,
            "distanceToBorder": -5,
        }

        # weights will be configured in game_start once agent_index is available
        self.weights = {}

        # tracking for exploration and stuck detection
        self._last_food_eaten = 0
        self._turns_since_food = 0
        self._position_visits = {}
        self._state_processed = False

    def game_start(self, initial_state):
        super().game_start(initial_state)
        self._distances.compute(initial_state.board)

        # choose initial weights based on fixed role parity now that agent_index is set
        # agents 0,1 are the first team; agents 2,3 are the second team
        # so we check agent_index % 4 < 2 to get first agent on each team as offense
        if (self.agent_index % 4) < 2:
            self.weights = self.offensive_weights.copy()
        else:
            self.weights = self.defensive_weights.copy()

        # remember food for carry calculations
        self._start_food = initial_state.food_count(agent_index=self.agent_index)

        # reset tracking variables
        self._last_food_eaten = 0
        self._turns_since_food = 0
        self._position_visits = {}
        self._state_processed = False

        # compute border positions
        board = initial_state.board
        mid = board.width // 2
        my_pos = initial_state.get_agent_position(self.agent_index)
        if my_pos and my_pos.col < mid:
            border_col = mid - 1
        else:
            border_col = mid

        self._border_positions = [
            Position(row, border_col)
            for row in range(board.height)
            if not board.is_wall(Position(row, border_col))
        ]

    def _determine_role(self, state) -> bool:
        """Return True for offense, False for defense this turn.

        Priority order:
        1. Opponents scared → offense (both agents capitalize on power pellet).
        2. If teammate is dead → defense (surviving agent switches to defense).
        3. Score-based strategy:
           - If score < -20 (losing badly): both agents offense (high risk).
           - Otherwise: dynamic role assignment based on distances.
        """
        my_index = self.agent_index

        # condition 1: scared opponents
        scared = state.get_scared_opponent_positions(agent_index=my_index)
        if scared:
            return True

        # condition 2: if teammate is dead, switch to defense
        teammate_index = my_index ^ 1
        if state.get_agent_position(teammate_index) is None:
            return False

        # Dynamic role assignment
        teammate_index = my_index ^ 1
        my_pos = state.get_agent_position(my_index)
        teammate_pos = state.get_agent_position(teammate_index)

        if my_pos is None or teammate_pos is None:
            # Fallback to parity if position unknown
            return (my_index % 4) < 2

        # Get enemy food positions
        enemy_food = state.get_food(agent_index=my_index)
        if enemy_food:
            my_food_distances = [
                self._distances.get_distance(my_pos, food_pos)
                for food_pos in enemy_food
            ]
            my_food_distances = [d for d in my_food_distances if d is not None]
            my_min_food_dist = (
                min(my_food_distances) if my_food_distances else float("inf")
            )

            teammate_food_distances = [
                self._distances.get_distance(teammate_pos, food_pos)
                for food_pos in enemy_food
            ]
            teammate_food_distances = [
                d for d in teammate_food_distances if d is not None
            ]
            teammate_min_food_dist = (
                min(teammate_food_distances)
                if teammate_food_distances
                else float("inf")
            )

            # If stuck (haven't eaten in many turns), penalize distance to switch roles
            if self._turns_since_food > 20:
                my_min_food_dist = float("inf")

            offensive_agent = (
                my_index
                if my_min_food_dist <= teammate_min_food_dist
                else teammate_index
            )

        # For defense, if invaders exist, assign closest agent
        invaders = state.get_invader_positions(agent_index=my_index)
        if invaders:
            my_invader_distances = [
                self._distances.get_distance(my_pos, inv_pos)
                for inv_pos in invaders.values()
            ]
            my_invader_distances = [d for d in my_invader_distances if d is not None]
            my_min_invader_dist = (
                min(my_invader_distances) if my_invader_distances else float("inf")
            )

            teammate_invader_distances = [
                self._distances.get_distance(teammate_pos, inv_pos)
                for inv_pos in invaders.values()
            ]
            teammate_invader_distances = [
                d for d in teammate_invader_distances if d is not None
            ]
            teammate_min_invader_dist = (
                min(teammate_invader_distances)
                if teammate_invader_distances
                else float("inf")
            )

            defensive_agent = (
                my_index
                if my_min_invader_dist <= teammate_min_invader_dist
                else teammate_index
            )
        else:
            # No invaders, assign the agent farther from food as defense
            if my_min_food_dist > teammate_min_food_dist:
                defensive_agent = my_index
            elif teammate_min_food_dist > my_min_food_dist:
                defensive_agent = teammate_index
            else:
                # Equal, use parity
                defensive_agent = teammate_index if (my_index % 4) < 2 else my_index

        # Return True if this agent is assigned to offense
        return my_index == offensive_agent

    def choose_action(self, state):
        """Override to track exploration and update stuck detection."""
        # Update tracking once per turn
        if not self._state_processed:
            self._state_processed = True
            current_food_eaten = self._start_food - state.food_count(
                agent_index=self.agent_index
            )
            if current_food_eaten > self._last_food_eaten:
                self._turns_since_food = 0
                self._last_food_eaten = current_food_eaten
            else:
                self._turns_since_food += 1

        # Use greedy features to choose action
        chosen_action = super().choose_action(state)

        # Update position visits and reset state processing
        successor = state.generate_successor(self.agent_index, chosen_action)
        successor_pos = successor.get_agent_position(self.agent_index)
        if successor_pos:
            self._position_visits[successor_pos] = (
                self._position_visits.get(successor_pos, 0) + 1
            )
        self._state_processed = False
        return chosen_action

    def compute_features(self, state, action):
        features = {}
        successor = state.generate_successor(self.agent_index, action)
        my_index = self.agent_index
        my_pos = successor.get_agent_position(my_index)

        if my_pos is None:
            return features

        play_offense = self._determine_role(successor)

        if play_offense:
            self.weights = self.offensive_weights.copy()

            # update tracking once per state
            if not self._state_processed:
                self._state_processed = True
                current_food_eaten = self._start_food - state.food_count(
                    agent_index=my_index
                )
                if current_food_eaten > self._last_food_eaten:
                    self._turns_since_food = 0
                    self._last_food_eaten = current_food_eaten
                else:
                    self._turns_since_food += 1

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

            # 3. Distance to opponents (non-scared) if we are on offense.
            opponents = successor.get_nonscared_opponent_positions(agent_index=my_index)
            opponent_positions = [pos for pos in opponents.values() if pos is not None]

            enemy_distances = [
                self._distances.get_distance(my_pos, opp)
                for opp in opponent_positions
                if opp is not None
            ]
            enemy_distances = [d for d in enemy_distances if d is not None]

            if enemy_distances:
                min_enemy_distance = min(enemy_distances)
                # Larger distance is better.
                features["enemyDistance"] = min_enemy_distance
                # Punish for being too close.
                features["enemyTooClose"] = 1 if min_enemy_distance <= 3 else 0
            else:
                # No visible enemies.
                features["enemyDistance"] = 20
                features["enemyTooClose"] = 0

            # 4. Carry-and-return: the more food we're carrying, the more
            #    we want to head home so we don't lose it all to a ghost.
            current_food = successor.food_count(agent_index=my_index)
            carrying = self._start_food - current_food

            home_distances = [
                self._distances.get_distance(my_pos, bp)
                for bp in self._border_positions
            ]
            home_distances = [d for d in home_distances if d is not None]
            nearest_home = min(home_distances) if home_distances else 0

            # Multiplied together so carrying 0 = no pull toward home.
            features["carryHome"] = carrying * nearest_home

            # 5. Distance to power capsules if being chased.
            capsules = successor.get_capsules(agent_index=my_index)
            if capsules and min_enemy_distance <= 5:
                # Being chased, prioritize capsules for temporary invincibility
                capsule_distances = [
                    self._distances.get_distance(my_pos, capsule)
                    for capsule in capsules
                ]
                capsule_distances = [d for d in capsule_distances if d is not None]
                features["distanceToCapsule"] = (
                    min(capsule_distances) if capsule_distances else 0
                )
            else:
                # Not being chased, ignore capsules
                features["distanceToCapsule"] = 0

        else:
            self.weights = self.defensive_weights.copy()

            # 1. On defense? True if agent is not a Pac-Man (i.e., on our side).
            features["onDefense"] = 1 if not successor.is_pacman(my_index) else 0

            # 2. Find invaders on our side using the capture helper.
            invader_positions = successor.get_invader_positions(agent_index=my_index)

            features["numInvaders"] = len(invader_positions)

            # 3. Distance to nearest invader.
            if invader_positions:
                distances = [
                    self._distances.get_distance(my_pos, inv_pos)
                    for inv_pos in invader_positions.values()
                ]
                distances = [d for d in distances if d is not None]
                features["invaderDistance"] = min(distances) if distances else 0
            else:
                features["invaderDistance"] = 0

            # Patrol near border when no invaders.
            border_distances = [
                self._distances.get_distance(my_pos, bp)
                for bp in self._border_positions
            ]
            border_distances = [d for d in border_distances if d is not None]
            features["distanceToBorder"] = (
                min(border_distances) if border_distances else 0
            )

            # 4. How close are invaders to our remaining food?
            if invader_positions:
                invader_indices = list(invader_positions.keys())
                first_invader_index = invader_indices[0]
                our_food_positions = successor.get_food(agent_index=first_invader_index)

                if our_food_positions:
                    distances_to_food = []
                    for inv_pos in invader_positions.values():
                        for food_pos in our_food_positions:
                            d = self._distances.get_distance(inv_pos, food_pos)
                            if d is not None:
                                distances_to_food.append(d)

                    distances_to_food = [d for d in distances_to_food if d is not None]
                    features["invaderFoodDistance"] = (
                        min(distances_to_food) if distances_to_food else 0
                    )
                else:
                    features["invaderFoodDistance"] = 0
            else:
                features["invaderFoodDistance"] = 0

        # Avoid stopping and reversing.
        agent_actions = state.get_agent_actions(my_index)

        if len(agent_actions) > 1:
            previous_action = agent_actions[-2]
            reverse_action = state.get_reverse_action(previous_action)
            features["reverse"] = 1 if action == reverse_action else 0
        else:
            features["reverse"] = 0

        features["stopped"] = 1 if action == pacai.core.action.STOP else 0

        return features
