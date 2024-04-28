import copy
import itertools
import random
import time

# fmt: off
state = [
    ['W', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', 'L'],
    ['W', '-', '-', '-', '-', '-', '-', 'B', 'B', 'B', '-', '-', '-', '-', '-', '-', '-', 'L'],
    ['W', '-', '-', '-', '-', '-', '-', 'B', 'B', 'B', '-', '-', '-', 'T', 'T', 'T', '-', '-'],
    ['W', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', 'T', 'T', 'T', '-', '-'],
    ['W', 'C', 'C', 'C', '-', 'V', 'V', 'V', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
    ['-', 'C', 'C', 'C', '-', 'V', 'V', 'V', '-', '-', '-', '-', '-', '-', '-', '-', 'U', 'U'],
    ['-', 'C', 'C', 'C', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', 'U', 'U'],
    ['-', '-', '-', '-', '-', '-', '-', 'P', 'P', '-', '-', '-', 'K', 'K', '-', '-', '-', '-'],
    ['-', '-', '-', '-', '-', '-', '-', 'P', 'P', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
    ['-', '-', '-', '-', '-', '-', '-', 'P', 'P', '-', '-', '-', '-', '-', '-', '-', 'Q', 'Q'],
    ['-', '-', '-', '-', '-', '-', '-', 'P', 'P', '-', '-', '-', 'G', 'G', '-', '-', 'Q', 'Q'],
    ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', 'G', 'G', '-', '-', '-', '-'],
    ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', 'G', 'G', '-', '-', '-', '-'],
    ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', 'G', 'G', '-', '-', '-', '-'],
    ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', 'G', 'G', 'M', 'M', '-', '-'],
    ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', 'F', 'F', 'F', 'F', 'F', 'F', '-'],
    ['A', 'A', 'A', 'A', 'A', '-', '-', '-', '-', '-', '-', 'F', 'F', 'F', 'F', 'F', 'F', '-'],
    ['A', 'A', 'A', 'A', 'A', '-', '-', '-', '-', '-', '-', '-', '-', '-', 'D', 'D', '-', '-'],
    ['A', 'A', 'A', 'A', 'A', '-', '-', '-', '-', '-', '-', '-', '-', '-', 'D', 'D', '-', '-'],
    ['E', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', 'D', 'D', '-', '-'],
]

target_state = [
    ['A', 'A', 'A', 'A', 'A', 'C', 'C', 'C', 'F', 'F', 'F', 'F', 'F', 'F', 'U', 'U', 'Q', 'Q'],
    ['A', 'A', 'A', 'A', 'A', 'C', 'C', 'C', 'F', 'F', 'F', 'F', 'F', 'F', 'U', 'U', 'Q', 'Q'],
    ['A', 'A', 'A', 'A', 'A', 'C', 'C', 'C', 'M', 'M', 'G', 'G', 'P', 'P', '-', '-', '-', '-'], 
    ['K', 'K', 'E', 'L', 'D', 'D', 'W', 'V', 'V', 'V', 'G', 'G', 'P', 'P', '-', '-', '-', '-'],
    ['B', 'B', 'B', 'L', 'D', 'D', 'W', 'V', 'V', 'V', 'G', 'G', 'P', 'P', '-', '-', '-', '-'],
    ['B', 'B', 'B', '-', 'D', 'D', 'W', 'T', 'T', 'T', 'G', 'G', 'P', 'P', '-', '-', '-', '-'],
    ['-', '-', '-', '-', '-', '-', 'W', 'T', 'T', 'T', 'G', 'G', '-', '-', '-', '-', '-', '-'],
    ['-', '-', '-', '-', '-', '-', 'W', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
    ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
    ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
    ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
    ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
    ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
    ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
    ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
    ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
    ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
    ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
    ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
    ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
]
# fmt: on

# ------------------ CLASS ------------------


class Puzzle:
    DIRECTIONS = ["up", "down", "left", "right"]

    def __init__(self, state, target_state, pieces):
        self.state = state
        self.target_state = target_state
        self.pieces = pieces
        self.empty_info = self.create_empty_space_info()
        self.piece_sizes = self.create_piece_size_dict()
        self.piece_info = self.create_piece_info()
        self.dependency_graph = self.build_dependency_graph()
        self.order_of_pieces = self.topological_sort()
        self.move_count = 0
        self.move_register = []

    # -------------DICT FUNCTIONS----------------

    def create_piece_size_dict(self):
        def flood_fill(state, x, y, old_value, visited):
            if (
                x < 0
                or y < 0
                or x >= len(state)
                or y >= len(state[0])
                or state[x][y] != old_value
                or visited[x][y]
            ):
                return []
            visited[x][y] = True
            cells = [(x, y)]
            cells += flood_fill(state, x + 1, y, old_value, visited)
            cells += flood_fill(state, x - 1, y, old_value, visited)
            cells += flood_fill(state, x, y + 1, old_value, visited)
            cells += flood_fill(state, x, y - 1, old_value, visited)
            return cells

        piece_sizes = {}
        for piece in self.pieces:
            visited = [[False] * len(self.state[0]) for _ in range(len(self.state))]
            for i, row in enumerate(self.state):
                for j, cell in enumerate(row):
                    if cell == piece and not visited[i][j]:
                        cells = flood_fill(self.state, i, j, piece, visited)
                        min_i, min_j = min(cells, key=lambda x: (x[0], x[1]))
                        max_i, max_j = max(cells, key=lambda x: (x[0], x[1]))
                        size = (max_j - min_j + 1, max_i - min_i + 1)  # (width, height)
                        piece_sizes.setdefault(piece, []).append(size)
        return piece_sizes

    def create_piece_info(self):
        piece_info = {}
        for piece in self.pieces:
            target_positions = self.get_positions(self.target_state, piece)
            current_positions = self.get_positions(self.state, piece)
            movable_directions = self.get_movable_directions(self.state, piece)
            movable_directions_end = self.get_movable_directions(
                self.target_state, piece
            )
            target_zones = self.calculate_target_zones(
                current_positions, target_positions, self.piece_sizes[piece][0]
            )
            distance = self.calculate_distance(
                current_positions[0], target_positions[0]
            )
            # in_target_position = if the current position == target position
            in_target_position = current_positions == target_positions
            piece_info[piece] = {
                "size": self.piece_sizes[piece][0],
                "target_positions": target_positions,
                "positions": current_positions,
                "moveable_directions": movable_directions,
                "final_movable_directions": movable_directions_end,
                "target_zones": target_zones,
                "in_target_position": in_target_position,
                "distance_to_target": distance,
            }
        piece_info = self.identify_target_neighbors(piece_info)
        self.piece_info = piece_info  # Now, piece_info is fully constructed.
        for piece in piece_info:
            occupying_zones = self.get_occupying_zones(self.state, piece)
            piece_info[piece]["obstacle_to"] = occupying_zones
            obstacles = self.get_obstacles(
                self.state, piece, piece_info[piece]["target_zones"]
            )
            piece_info[piece]["obstacles"] = obstacles
            best_target_zone, best_target_zone_obstacles = self.find_best_target_zone(
                piece
            )
            piece_info[piece]["btz"] = best_target_zone
            piece_info[piece]["btz_obstacles"] = best_target_zone_obstacles
            potential_areas = self.find_potential_areas(piece)
            piece_info[piece]["potential_areas"] = potential_areas
            current_positions = piece_info[piece]["positions"]
            target_positions = piece_info[piece]["target_positions"]
            blocking_score = self.calculate_blocking_score(piece)
            alignment_score = self.calculate_alignment_score(piece)
            # a star path to target position?
            a_star_path = self.a_star_search(
                self.state, piece, piece_info[piece]["target_positions"]
            )
            if a_star_path:
                piece_info[piece]["a_star_path"] = self.path_to_moves(
                    piece, a_star_path
                )
            fitness = self.calculate_fitness(
                piece,
                current_positions,
                target_positions,
                blocking_score,
                alignment_score,
            )
            piece_info[piece]["fitness"] = fitness
        return piece_info

    def get_movable_directions(self, state, piece):
        movable_directions = {}
        positions = self.get_positions(state, piece)
        min_row = min(pos[0] for pos in positions)
        max_row = max(pos[0] for pos in positions)
        min_col = min(pos[1] for pos in positions)
        max_col = max(pos[1] for pos in positions)
        # Check how far the piece can move up
        distance = 0
        while min_row - distance - 1 >= 0 and all(
            state[min_row - distance - 1][pos[1]] == "-"
            for pos in positions
            if pos[0] == min_row
        ):
            distance += 1
        if distance > 0:
            movable_directions["up"] = distance
        # Check how far the piece can move down
        distance = 0
        while max_row + distance + 1 < len(state) and all(
            state[max_row + distance + 1][pos[1]] == "-"
            for pos in positions
            if pos[0] == max_row
        ):
            distance += 1
        if distance > 0:
            movable_directions["down"] = distance
        # Check how far the piece can move left
        distance = 0
        while min_col - distance - 1 >= 0 and all(
            state[pos[0]][min_col - distance - 1] == "-"
            for pos in positions
            if pos[1] == min_col
        ):
            distance += 1
        if distance > 0:
            movable_directions["left"] = distance
        # Check how far the piece can move right
        distance = 0
        while max_col + distance + 1 < len(state[0]) and all(
            state[pos[0]][max_col + distance + 1] == "-"
            for pos in positions
            if pos[1] == max_col
        ):
            distance += 1
        if distance > 0:
            movable_directions["right"] = distance
        return movable_directions

    def identify_target_neighbors(self, piece_info):
        target_state = self.target_state
        directions = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
        for piece in self.pieces:
            neighbors = {}
            target_positions = piece_info[piece]["target_positions"]
            for direction, (dx, dy) in directions.items():
                for x, y in target_positions:
                    nx, ny = x + dx, y + dy
                    if (
                        0 <= nx < len(target_state)
                        and 0 <= ny < len(target_state[0])
                        and target_state[nx][ny] != "-"
                    ):
                        neighbor_piece = target_state[nx][ny]
                        if neighbor_piece not in neighbors:
                            neighbors[neighbor_piece] = [direction]
                        else:
                            neighbors[neighbor_piece].append(direction)
            piece_info[piece]["target_neighbors"] = neighbors
        return piece_info

    def get_positions(self, state, piece):
        positions = []
        for i, row in enumerate(state):
            for j, cell in enumerate(row):
                if cell == piece:
                    positions.append((i, j))
        return positions

    # -------------FITNESS FUNCTIONS----------------

    def calculate_blocking_score(self, piece):
        blocking_score = 0
        # Check if the piece is blocking the target zone of another piece
        for other_piece in self.pieces:
            if other_piece != piece:
                target_zones = self.piece_info[other_piece]["target_zones"]
                if any(
                    self.is_in_target_zones(self.state, piece, zone)
                    for zone in target_zones
                ):
                    blocking_score += 1
        return blocking_score

    def calculate_alignment_score(self, piece):
        alignment_score = 0
        current_positions = self.get_positions(self.state, piece)
        target_positions = self.piece_info[piece]["target_positions"]
        for (cur_row, cur_col), (tar_row, tar_col) in zip(
            current_positions, target_positions
        ):
            alignment_score += abs(tar_row - cur_row) + abs(tar_col - cur_col)
        return alignment_score

    def calculate_fitness(
        self,
        piece,
        current_positions,
        target_positions,
        blocking_score,
        alignment_score,
    ):
        total_distance = 0
        for current_position, target_position in zip(
            current_positions, target_positions
        ):
            total_distance += self.calculate_distance(current_position, target_position)
        return total_distance + blocking_score - alignment_score

    def calculate_state_fitness(self):
        total_fitness = 0
        for piece in self.pieces:
            total_fitness += self.calculate_fitness(piece)
        alignment_score = self.calculate_alignment_score()
        return total_fitness - alignment_score

    # -------------SPACE FUNCTIONS----------------

    @staticmethod
    def calculate_distance(current_position, target_position):
        # distance is rows across, columns down
        return abs(current_position[0] - target_position[0]) + abs(
            current_position[1] - target_position[1]
        )

    def create_empty_space_info(self):
        state = self.state
        empty_space_positions = self.get_positions(state, "-")
        empty_space_info = {
            "size": (1, 1),  # Since dashes represent single empty cells
            "positions": empty_space_positions,
            "moveable_directions": [],  # Empty spaces don't move, so no movable directions
        }
        return {"-": empty_space_info}

    def create_empty_space_info(self):
        state = self.state
        empty_space_positions = self.get_positions(state, "-")
        empty_space_info = {
            "size": (1, 1),  # Since dashes represent single empty cells
            "positions": empty_space_positions,
        }
        return {"-": empty_space_info}

    def calculate_target_zones(self, current_positions, target_positions, piece_size):
        zones = []

        # Extract width and height from piece_size
        piece_width, piece_height = piece_size

        # Calculate the minimum rows and columns for the current and target positions
        current_min_row = min(pos[0] for pos in current_positions)
        current_min_col = min(pos[1] for pos in current_positions)
        target_min_row = min(pos[0] for pos in target_positions)
        target_min_col = min(pos[1] for pos in target_positions)

        # Determine the direction of movement (up/down and left/right)
        vertical_direction = 1 if target_min_row > current_min_row else -1
        horizontal_direction = 1 if target_min_col > current_min_col else -1

        # Calculate the zones
        for order in [
            (0, 1),
            (1, 0),
        ]:  # (0, 1) for rows first, (1, 0) for columns first
            target_zone = set(current_positions)
            row, col = current_min_row, current_min_col

            for axis in order:
                # Move vertically
                if axis == 0:
                    while row != target_min_row:
                        for col_offset in range(piece_width):
                            for row_offset in range(
                                piece_height
                            ):  # Consider the entire height while moving
                                target_zone.add((row + row_offset, col + col_offset))
                        row += vertical_direction
                # Move horizontally
                else:
                    while col != target_min_col:
                        for row_offset in range(piece_height):
                            for col_offset in range(
                                piece_width
                            ):  # Consider the entire width while moving
                                target_zone.add((row + row_offset, col + col_offset))
                        col += horizontal_direction

            # Adding the final target position
            for row_offset in range(piece_height):
                for col_offset in range(piece_width):
                    target_zone.add(
                        (target_min_row + row_offset, target_min_col + col_offset)
                    )

            zones.append(target_zone)

        return zones

    def build_dependency_graph(self):
        graph = {piece: [] for piece in self.piece_info}
        for piece, info in self.piece_info.items():
            target_neighbors = info["target_neighbors"]
            for neighbor_piece, directions in target_neighbors.items():
                # If the neighbor piece cannot move in the opposite direction in its final state, it must be placed first
                neighbor_final_movable_directions = self.piece_info[neighbor_piece][
                    "final_movable_directions"
                ]
                for direction in directions:
                    opposite_direction = {
                        "up": "down",
                        "down": "up",
                        "left": "right",
                        "right": "left",
                    }[direction]
                    if opposite_direction not in neighbor_final_movable_directions:
                        graph[piece].append(neighbor_piece)

            # Consider target zones and final movable directions as before
            if not info["final_movable_directions"]:
                for other_piece in self.piece_info:
                    if piece != other_piece:
                        graph[piece].append(other_piece)
            else:
                for other_piece, other_info in self.piece_info.items():
                    if piece != other_piece:
                        if any(
                            zone.intersection(other_info["positions"])
                            for zone in info["target_zones"]
                        ):
                            graph[piece].append(other_piece)
        return graph

    def topological_sort(self):
        graph = self.dependency_graph
        visited = {node: False for node in graph}
        stack = []
        for node in graph:
            if not visited[node]:
                self.dfs(node, graph, visited, stack)
        return stack[::-1]

    def dfs(self, node, graph, visited, stack):
        visited[node] = True
        for neighbor in graph[node]:
            if not visited[neighbor]:
                self.dfs(neighbor, graph, visited, stack)
        stack.append(node)

    def is_in_target_zones(self, state, piece, target_zone):
        piece_positions = self.get_positions(state, piece)
        return any(position in target_zone for position in piece_positions)

    def get_occupying_zones(self, state, piece):
        occupying_zones = []
        for other_piece in self.pieces:
            if other_piece != piece:
                target_zones = self.piece_info[other_piece]["target_zones"]
                if any(
                    self.is_in_target_zones(state, piece, zone) for zone in target_zones
                ):
                    occupying_zones.append(other_piece)
        return occupying_zones

    def get_obstacles(self, state, piece, target_zones):
        obstacles = []
        for other_piece in self.pieces:
            if other_piece != piece:
                if any(
                    self.is_in_target_zones(state, other_piece, zone)
                    for zone in target_zones
                ):
                    obstacles.append(other_piece)
        return obstacles

    def get_area(self, start_position, size):
        """Get a rectangular area of the specified size starting from start_position."""
        end_position = (
            start_position[0] + size[0] - 1,
            start_position[1] + size[1] - 1,
        )
        if end_position[0] >= len(self.state) or end_position[1] >= len(self.state[0]):
            return None
        return start_position, end_position

    def is_area_empty(self, area):
        """Check if the specified area is empty."""
        if not area:
            return False
        start, end = area
        for i in range(start[0], end[0] + 1):
            for j in range(start[1], end[1] + 1):
                if self.state[i][j] != "-":
                    return False
        return True

    def fits_in_area(self, piece, area_top_left, area_bottom_right):
        piece_width, piece_height = self.piece_info[piece]["size"]
        for i in range(area_bottom_right[0] - area_top_left[0] - piece_height + 2):
            for j in range(area_bottom_right[1] - area_top_left[1] - piece_width + 2):
                if all(
                    self.state[area_top_left[0] + i + x][area_top_left[1] + j + y]
                    in ["-", piece]
                    for x in range(piece_height)
                    for y in range(piece_width)
                ):
                    return True
        return False

    def find_potential_areas(self, piece):
        piece_width, piece_height = self.piece_info[piece]["size"]
        potential_areas = []

        for row in range(len(self.state) - piece_height + 1):
            for col in range(len(self.state[row]) - piece_width + 1):
                area_top_left = (row, col)
                area_bottom_right = (row + piece_height - 1, col + piece_width - 1)
                if self.fits_in_area(piece, area_top_left, area_bottom_right):
                    potential_areas.append(
                        {
                            (area_top_left[0] + i, area_top_left[1] + j)
                            for i in range(piece_height)
                            for j in range(piece_width)
                        }
                    )

        return potential_areas

    def find_best_target_zone(self, piece):
        # Find the target zone with the least number of obstacles
        target_zones = self.piece_info[piece]["target_zones"]
        best_target_zone = None
        min_obstacles = float("inf")
        for target_zone in target_zones:
            obstacles = self.get_obstacles(self.state, piece, [target_zone])
            if len(obstacles) < min_obstacles:
                best_target_zone = target_zone
                min_obstacles = len(obstacles)
                obs = obstacles
        return best_target_zone, obs

    # -------------MOVE FUNCTIONS----------------

    def apply_move(self, piece, direction, distance):
        if self.is_valid_move(piece, direction, distance):
            new_state = copy.deepcopy(self.state)
            positions = self.piece_info[piece]["positions"]
            overlaps = False
            if positions:
                if direction == "up":
                    for pos in positions:
                        for d in range(1, distance + 1):
                            if new_state[pos[0] - d][pos[1]] != "-":
                                overlaps = True
                        new_state[pos[0]][pos[1]] = "-"
                        new_state[pos[0] - distance][pos[1]] = piece
                elif direction == "down":
                    for pos in sorted(positions, reverse=True):
                        for d in range(1, distance + 1):
                            if new_state[pos[0] + d][pos[1]] != "-":
                                overlaps = True
                        new_state[pos[0]][pos[1]] = "-"
                        new_state[pos[0] + distance][pos[1]] = piece
                elif direction == "left":
                    for pos in positions:
                        for d in range(1, distance + 1):
                            if new_state[pos[0]][pos[1] - d] != "-":
                                overlaps = True
                        new_state[pos[0]][pos[1]] = "-"
                        new_state[pos[0]][pos[1] - distance] = piece
                elif direction == "right":
                    for pos in sorted(positions, key=lambda x: x[1], reverse=True):
                        for d in range(1, distance + 1):
                            if new_state[pos[0]][pos[1] + d] != "-":
                                overlaps = True
                        new_state[pos[0]][pos[1]] = "-"
                        new_state[pos[0]][pos[1] + distance] = piece
            if overlaps:
                return self.state
            self.piece_info[piece]["positions"] = self.get_positions(new_state, piece)
            self.state = new_state
            if self.is_valid_state():
                self.piece_info = self.create_piece_info()
                self.move_count += 1
                self.move_register.append((piece, direction, distance))
                Display.display_state(self.state)
                return self.state
        return self.state

    def is_valid_state(self):
        new_piece_sizes = self.create_piece_size_dict()
        for piece in self.piece_sizes:
            if sorted(self.piece_sizes[piece]) != sorted(
                new_piece_sizes.get(piece, [])
            ):
                return False
        return True

    def is_valid_move(self, piece, direction, distance):
        positions = self.piece_info[piece]["positions"]
        if direction == "up":
            return all(pos[0] >= distance for pos in positions)
        elif direction == "down":
            return all(pos[0] < len(self.state) - distance for pos in positions)
        elif direction == "left":
            return all(pos[1] >= distance for pos in positions)
        elif direction == "right":
            return all(pos[1] < len(self.state[0]) - distance for pos in positions)

    def calculate_moves(self, piece, area):
        min_row = min([pos[0] for pos in area])
        min_col = min([pos[1] for pos in area])
        current_positions = self.piece_info[piece]["positions"]
        return max(
            abs(current_positions[0][0] - min_row),
            abs(current_positions[0][1] - min_col),
        )

    def determine_move_direction_and_distance(self, piece, target_zone):
        min_row = min([pos[0] for pos in target_zone])
        min_col = min([pos[1] for pos in target_zone])
        # Calculate row and column difference
        delta_row = min_row - self.piece_info[piece]["positions"][0][0]
        delta_col = min_col - self.piece_info[piece]["positions"][0][1]
        # Determine direction
        # Vertical
        if delta_row > 0:
            vertical_direction = "down"
        elif delta_row < 0:
            vertical_direction = "up"
        else:
            vertical_direction = None
        # Horizontal
        if delta_col > 0:
            horizontal_direction = "right"
        elif delta_col < 0:
            horizontal_direction = "left"
        else:
            horizontal_direction = None
        return vertical_direction, abs(delta_row), horizontal_direction, abs(delta_col)

    @staticmethod
    def heuristic(piece_positions, end_positions):
        return sum(
            abs(p[0] - e[0]) + abs(p[1] - e[1])
            for p, e in zip(piece_positions, end_positions)
        )

    def get_neighbors(self, piece_positions, board):
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        valid_moves = []

        for dx, dy in directions:
            new_piece = tuple((p[0] + dx, p[1] + dy) for p in piece_positions)
            if all(
                0 <= p[0] < len(board)
                and 0 <= p[1] < len(board[0])
                and (board[p[0]][p[1]] == 0 or board[p[0]][p[1]] == 2)
                for p in new_piece
            ):
                valid_moves.append(tuple(new_piece))

        return valid_moves

    def a_star_search(self, state, piece, target_area):
        board = [[0 for _ in range(len(state[0]))] for _ in range(len(state))]
        piece_positions = self.get_positions(state, piece)
        for i, row in enumerate(state):
            for j, cell in enumerate(row):
                if cell != "-" and cell != piece:
                    board[i][j] = 1
                if (i, j) in piece_positions:
                    board[i][j] = 2
        start = tuple(piece_positions)
        end = tuple(target_area)

        open_list = [start]
        closed_list = set()
        g = {start: 0}
        h = {start: self.heuristic(start, end)}
        f = {start: h[start]}
        parents = {}

        while open_list:
            current = min(open_list, key=lambda position: f[position])
            if current == end:
                path = []
                while current in parents:
                    path.append(current)
                    current = parents[current]
                path.append(start)
                path.reverse()
                return path

            open_list.remove(current)
            closed_list.add(current)

            for neighbor in self.get_neighbors(current, board):
                if neighbor in closed_list:
                    continue
                tentative_g = g[current] + 1
                if neighbor not in open_list or tentative_g < g.get(
                    neighbor, float("inf")
                ):
                    parents[neighbor] = current
                    g[neighbor] = tentative_g
                    h[neighbor] = self.heuristic(neighbor, end)
                    f[neighbor] = g[neighbor] + h[neighbor]
                    open_list.append(neighbor)
        return []

    def path_to_moves(self, piece, path):
        if not path:
            return None
        if len(path) == 1:
            return ["Target reached"]
        moves = []
        for i in range(1, len(path)):
            current_pos, next_pos = (
                path[i - 1][0],
                path[i][0],
            )  # considering only the first coordinate of each piece for simplicity
            if next_pos[0] < current_pos[0]:
                moves.append("up")
            elif next_pos[0] > current_pos[0]:
                moves.append("down")
            elif next_pos[1] < current_pos[1]:
                moves.append("left")
            elif next_pos[1] > current_pos[1]:
                moves.append("right")

        # Combine consecutive moves in the same direction
        combined_moves = []
        current_move = moves[0]
        count = 1
        for i in range(1, len(moves)):
            if moves[i] == current_move:
                count += 1
            else:
                combined_moves.append((piece, current_move, count))
                current_move = moves[i]
                count = 1
        combined_moves.append((piece, current_move, count))

        return combined_moves

    def move_piece_to_target(self, piece, target_area):
        vd, vl, hd, hl = self.determine_move_direction_and_distance(piece, target_area)
        if vd:
            self.apply_move(piece=piece, direction=vd, distance=vl)
        if hd:
            self.apply_move(piece=piece, direction=hd, distance=hl)

    # -------------EXPLORATORY FUNCTIONS----------------

    def removal_hypothesis(self, piece, obstacles):
        results = []
        solution_found = False
        min_obstacles = (
            len(obstacles) + 1
        )  # Initialize with a value greater than the maximum possible obstacles

        for r in range(1, len(obstacles) + 1):
            if solution_found and r > min_obstacles:
                break

            for subset in itertools.combinations(obstacles, r):
                # Temporarily set the obstacles in the subset to '-'
                original_states = {}
                for obs in subset:
                    original_states[obs] = []
                    for pos in self.piece_info[obs]["positions"]:
                        original_states[obs].append(self.state[pos[0]][pos[1]])
                        self.state[pos[0]][pos[1]] = "-"
                # Check if there is an a_star_path
                path = self.a_star_search(
                    self.state, piece, self.piece_info[piece]["target_positions"]
                )
                # Convert the obstacles back to their original state
                for obs, states in original_states.items():
                    for pos, state in zip(self.piece_info[obs]["positions"], states):
                        self.state[pos[0]][pos[1]] = state

                # Add the result to the results list
                result = {
                    "obstacles": list(subset),
                    "a_star_path_length": len(path) if path else 0,
                    "a_star_path": self.path_to_moves(piece, path) if path else False,
                }
                results.append(result)

                # If a path is found, update min_obstacles and mark solution_found as True
                if path:
                    solution_found = True
                    min_obstacles = min(min_obstacles, len(subset))

        # If no solutions were found, return an empty list
        if not any(res["a_star_path"] for res in results):
            return []

        # Filter results to include only the solutions with the fewest obstacles
        results = [res for res in results if len(res["obstacles"]) == min_obstacles]
        return results

    # -------------OBSTACLE FUNCTIONS----------------

    def create_pa_dict(self, piece, obstacle, pa_new):
        current_positions = self.get_positions(self.state, obstacle)
        PA_dict = {}
        area_id = 0  # Initialize the unique ID
        for area in pa_new:
            # check if there is an a_star_path
            PO_count = self.determine_PO_count(piece, area, 3)
            sorted_area = tuple(sorted(area, key=lambda x: (x[0], x[1])))
            path = self.a_star_search(self.state, obstacle, sorted_area)
            path = self.path_to_moves(obstacle, path)
            if not path:
                break
            else:
                vl = sum(move[2] for move in path if move[1] in ["up", "down"])
                hl = sum(move[2] for move in path if move[1] in ["left", "right"])
                obstacles_1 = []
                obstacles_2 = []
                temp_target_zones = self.a_star_zones(path, obstacle)
            PA_dict[area_id] = {
                "area": area,
                "temp_target_zones": temp_target_zones,
                "obstacles_path_1": obstacles_1,
                "obstacles_path_2": obstacles_2,
                "distance_to_area": vl + hl,
                "PO_count": PO_count,
                "path": path,
                "fitness": (len(obstacles_1) + len(obstacles_2)) * 2
                + vl
                + hl
                + PO_count * 5,
            }
            area_id += 1  # Increment the unique ID for the next area
        return PA_dict

    def a_star_zones(self, a_star_path, piece):
        zones = []
        piece_size = self.piece_info[piece]["size"]
        # Extract width and height from piece_size
        piece_width, piece_height = piece_size
        # Start with the current position of the piece
        current_positions = set(self.piece_info[piece]["positions"])

        # Helper function to generate the zone given the top-left corner
        def generate_zone(top_left):
            current_min_row, current_min_col = top_left
            zone = set()
            for row_offset in range(piece_height):
                for col_offset in range(piece_width):
                    zone.add(
                        (current_min_row + row_offset, current_min_col + col_offset)
                    )
            return zone

        # Iterate over the moves in the a_star_path to update current_positions
        for move in a_star_path:
            try:
                _, direction, distance = move
                for _ in range(distance):  # Move one step at a time
                    if direction == "up":
                        current_positions = {
                            (pos[0] - 1, pos[1]) for pos in current_positions
                        }
                    elif direction == "down":
                        current_positions = {
                            (pos[0] + 1, pos[1]) for pos in current_positions
                        }
                    elif direction == "left":
                        current_positions = {
                            (pos[0], pos[1] - 1) for pos in current_positions
                        }
                    elif direction == "right":
                        current_positions = {
                            (pos[0], pos[1] + 1) for pos in current_positions
                        }
                    zones.append(
                        generate_zone(
                            (
                                min(pos[0] for pos in current_positions),
                                min(pos[1] for pos in current_positions),
                            )
                        )
                    )
            except:
                return None
        return zones

    def determine_PO_count(self, piece, area, index_offset):
        count = 0
        for offset in range(1, index_offset + 1):
            next_piece = self.order_of_pieces[
                self.order_of_pieces.index(piece) + offset
            ]
            next_piece_target_zones = self.piece_info[next_piece]["target_zones"]
            if any(area.intersection(zone) for zone in next_piece_target_zones):
                # more for sooner, less for later
                count += index_offset * 2 - offset
        return count

    def create_obstacle_dict(self, piece):
        obstacles = self.piece_info[piece]["obstacles"]
        obstacles = [
            i for i in obstacles if self.piece_info[i]["in_target_position"] == False
        ]
        results = self.removal_hypothesis(piece, obstacles)
        results = [res for res in results if res["a_star_path"]]
        for result in results:
            obstacle_dict = {}
            obs = result["obstacles"]
            PZ = self.a_star_zones(result["a_star_path"], piece)
            for obstacle in obs:
                PA = self.piece_info[obstacle]["potential_areas"]
                size = self.piece_info[obstacle]["size"]
                target_positions = self.piece_info[obstacle]["target_positions"]
                distance_to_target = self.piece_info[obstacle]["distance_to_target"]
                PA_new = [
                    area
                    for area in PA
                    if all(
                        len(area.intersection(target_zone)) == 0 for target_zone in PZ
                    )
                ]
                current_positions = self.get_positions(self.state, obstacle)
                PA_new = sorted(
                    PA_new,
                    key=lambda x: self.calculate_distance(
                        current_positions[0],
                        (min([pos[0] for pos in x]), min([pos[1] for pos in x])),
                    ),
                )
                # get area with best fitness
                PA_dict = self.create_pa_dict(piece, obstacle, PA_new)
                if PA_dict:
                    best_area = PA_dict[
                        min(PA_dict, key=lambda x: PA_dict[x]["fitness"])
                    ]["area"]
                    distance_to_area = PA_dict[
                        min(PA_dict, key=lambda x: PA_dict[x]["fitness"])
                    ]["distance_to_area"]
                    ob_path = PA_dict[
                        min(PA_dict, key=lambda x: PA_dict[x]["fitness"])
                    ]["path"]
                    fitness = PA_dict[
                        min(PA_dict, key=lambda x: PA_dict[x]["fitness"])
                    ]["fitness"]
                else:
                    best_area = None
                    distance_to_area = None
                    ob_path = None
                    fitness = float("inf")
                obstacle_dict[obstacle] = {
                    "size": size,
                    "target_positions": target_positions,
                    "distance_to_target": distance_to_target,
                    "distance_to_area": distance_to_area,
                    "best_area": best_area,
                    "ob_path": ob_path,
                    "fitness": fitness,
                }
            result["obstacle_dict"] = obstacle_dict
            result["option_fitness"] = sum(
                [obstacle_dict[obstacle]["fitness"] for obstacle in obstacle_dict]
            )
        return results


class Display:

    @staticmethod
    def display_state(state):
        for row in state:
            print("".join(row))
        print()

    @staticmethod
    def display_target_zones(state, target_zones):
        # Convert the state to a list of lists for easier manipulation
        grid = [list(row) for row in state]
        # Iterate through the target zones and mark them in the grid
        for zone in target_zones:
            for x, y in zone:
                if 0 <= x < len(grid) and 0 <= y < len(
                    grid[0]
                ):  # Validate the coordinates
                    grid[x][y] = "*"
                else:
                    print(f"Invalid coordinate: ({x}, {y})")
        for row in grid:
            print("".join(row))

    @staticmethod
    def display_paths_and_obstacles(state, paths, obstacles):
        # Convert the state to a list of lists for easier manipulation
        grid = [list(row) for row in state]
        # Iterate through the paths and mark them in the grid
        for path in paths:
            for x, y in path:
                grid[x][y] = "P"  # Marking the path cells with 'P'
        # Iterate through the obstacles and mark them in the grid
        for x, y in obstacles:
            grid[x][y] = "O"  # Marking the obstacle cells with 'O'
        # Convert the grid back to a string and return
        return "\n".join("".join(row) for row in grid)

    @staticmethod
    def display_board(board):
        for row in board:
            row = [str(cell) for cell in row]
            print("".join(row))
        print()


class Population:
    def __init__(self, puzzle, population_size=100):
        self.puzzle = puzzle
        self.population_size = population_size
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            sequence = []
            for piece in self.puzzle.order_of_pieces:
                obstacles = self.puzzle.piece_info[piece]["obstacles"]
                for obstacle in obstacles:
                    potential_areas = self.puzzle.piece_info[obstacle][
                        "potential_areas"
                    ]
                    area = random.choice(potential_areas)
                    # test new target state with obstacle in area, and previous obstacle piece target state as -


# fmt: off
DIRECTIONS = ['up', 'down', 'left', 'right']
PIECES = ['A', 'C', 'E', 'F', 'G', 'K','B', 'L', 'P', 'Q', 'T', 'U', 'V','M','D','W']
# fmt: on
POPULATION_SIZE = 100
TOURNAMENT_SIZE = 10
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.4
max_generations = 100

puzzle = Puzzle(state, target_state, PIECES)


def update_info(piece):
    INFO = puzzle.piece_info[piece]
    OB = INFO["obstacles"]
    BTZ = INFO["btz"]
    try:
        PATH = INFO["a_star_path"]
        if PATH:
            for move in PATH:
                puzzle.apply_move(*move)
            Display.display_state(puzzle.state)
            INFO, OB, BTZ = update_info(piece)
            if INFO["in_target_position"]:
                return INFO, OB, BTZ
    except:
        pass
    return INFO, OB, BTZ


def non_a_star(obstacle, piece):
    BTZ = puzzle.piece_info[piece]["btz"]
    PA = puzzle.piece_info[obstacle]["potential_areas"]
    PA_new = [
        area
        for area in PA
        if all(len(area.intersection(target_zone)) == 0 for target_zone in [BTZ])
    ]
    PA_new = sorted(
        PA_new,
        key=lambda x: puzzle.calculate_distance(
            INFO["positions"][0],
            (min([pos[0] for pos in x]), min([pos[1] for pos in x])),
        ),
    )
    PA_dict = puzzle.create_pa_dict(piece, obstacle, PA_new)
    if PA_dict:
        # choose a random option from the top 5
        top_5 = sorted(PA_dict, key=lambda x: PA_dict[x]["fitness"])[:5]
        random_area = PA_dict[random.choice(top_5)]["area"]
        path = puzzle.a_star_search(puzzle.state, obstacle, random_area)
        if path:
            for move in path:
                try:
                    puzzle.apply_move(*move)
                except:
                    non_a_star(obstacle, piece)


for piece in puzzle.order_of_pieces:
    INFO, OB, BTZ = update_info(piece)
    try_count = 0
    while OB != []:
        if try_count < 3:
            POD = puzzle.create_obstacle_dict(piece)
            if POD != []:
                best_option = random.choice(POD)
                obs = best_option["obstacles"]
                for obstacle in obs:
                    path = best_option["obstacle_dict"][obstacle]["ob_path"]
                    if path:
                        try:
                            for move in path:
                                puzzle.apply_move(*move)
                        except:
                            non_a_star(obstacle, piece)
                INFO, OB, BTZ = update_info(piece)
            else:
                for obstacle in OB:
                    non_a_star(obstacle, piece)
                    INFO, OB, BTZ = update_info(piece)
            try_count += 1
        else:
            # move a random piece not in its target position
            random_count = 0
            if random_count < 2:
                non_final_pieces = [
                    piece
                    for piece in puzzle.order_of_pieces
                    if puzzle.piece_info[piece]["in_target_position"] == False
                ]
                random_piece = random.choice(non_final_pieces)
                random_direction = random.choice(DIRECTIONS)
                random_distance = random.randint(1, 10)
                puzzle.apply_move(random_piece, random_direction, random_distance)
                INFO, OB, BTZ = update_info(piece)
    try:
        puzzle.move_piece_to_target(piece, INFO["target_positions"])
        INFO, OB, BTZ = update_info(piece)
        if INFO["in_target_position"]:
            continue
    except:
        PATH = INFO["a_star_path"]
        if PATH:
            for move in PATH:
                puzzle.apply_move(*move)
            INFO, OB, BTZ = update_info(piece)
            if INFO["in_target_position"]:
                continue

print(puzzle.move_count)
print(puzzle.move_register)
