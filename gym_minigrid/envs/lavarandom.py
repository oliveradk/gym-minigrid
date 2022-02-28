from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class LavaRandomEnv(MiniGridEnv):
    """
    Environment with same underlying dynamics as LavaGap env, but where the lava
    are pointed to by arrows on different tiles
    """

    def __init__(self, size, num_obstacles=2, show_arrows=False, seed=None):
        self.obstacle_type = HiddenLava if show_arrows else Lava
        self.num_obstacles = num_obstacles
        self.show_arrows = show_arrows
        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True,
            #agent_view_size=3,
            seed=seed
        )

    def _gen_grid(self, width, height):
        assert width >= 5 and height >= 5

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.agent_pos = (1, 1)
        self.agent_dir = 0

        # Place a goal square in the bottom-right corner
        self.goal_pos = np.array((width - 2, height - 2))
        self.put_obj(Goal(), *self.goal_pos)

        #generate and store lava locations
        self.lava_locs = []
        self._gen_lava(width, height)


        #generate arrows for each lava location
        if self.show_arrows:
            self.arrow_locs = []
            self._gen_arrows(width, height)

        self.mission = (
            "avoid the lava and get to the green goal square"
        )

    def _gen_lava(self, width, height):
        for i in range(self.num_obstacles):
            loc = self._rand_pos(xLow=1, xHigh=width-1, yLow=1, yHigh=height-1)
            while np.all(loc == self.agent_pos) or np.all(loc == self.goal_pos):
                loc = self._rand_pos(xLow=1, xHigh=width - 1, yLow=1, yHigh=height - 1)
            self.lava_locs.append(loc)
            self.put_obj(self.obstacle_type(), *loc)

    def _gen_arrows(self, width, height):
        for loc in self.lava_locs:
            arrow_loc, orient, dist = self._pick_arrrow_loc(width, height, loc)
            cur_tile = self.grid.get(*arrow_loc)
            while cur_tile.type == 'arrow':
                arrow_loc, orient, dist = self._pick_arrrow_loc(width, height, loc)
                cur_tile = self.grid.get(*arrow_loc)
            if dist == 1:
                color = 'red'
            elif dist == 2:
                color = 'yellow'
            else:
                color = 'purple'
            self.arrow_locs.append(arrow_loc)
            self.put_obj(Arrow(orientation=orient, color=color), *arrow_loc)

    def _pick_arrrow_loc(self, width, height, loc):
        edge = np.random.choice(list(Arrow.ORIENT_DICT.keys()))
        if edge == 'S':
            arrow_loc = np.array([loc[0], 0])
        elif edge == "N":
            arrow_loc = np.array([loc[0], height - 1])
        elif edge == "W":
            arrow_loc = np.array([width - 1, loc[1]])
        else:  # edge == 'E'
            arrow_loc = np.array([0, loc[1]])
        dist = int(np.linalg.norm(arrow_loc - np.array(loc)))
        return arrow_loc, edge, dist

class RandomAgentLavaRandomEnv(LavaRandomEnv):
    def _gen_grid(self, width, height):
        assert width >= 5 and height >= 5

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.goal_pos = np.array((width - 2, height - 2))
        self.put_obj(Goal(), *self.goal_pos)

        # Place the agent randomly
        self._set_agent_random()

        #generate and store lava locations
        self.lava_locs = []
        self._gen_lava(width, height)

        #generate arrows for each lava location
        if self.show_arrows:
            self.arrow_locs = []
            self._gen_arrows(width, height)

        self.mission = (
            "avoid the lava and get to the green goal square"
        )

    def _set_agent_random(self):
        pos = self.goal_pos
        while np.all(pos == self.goal_pos):
            pos = self._rand_pos(1, 3, 1, 3)
        self.agent_pos = pos
        self.agent_dir = self._rand_int(0, 4)


class LavaRandomS5Env(LavaRandomEnv):
    def __init__(self):
        super().__init__(size=5)

class LavaRandomS6Env(LavaRandomEnv):
    def __init__(self):
        super().__init__(size=6)

class LavaRandomS7Env(LavaRandomEnv):
    def __init__(self):
        super().__init__(size=7)

class ArrowsRandomS5Env(LavaRandomEnv):
    def __init__(self):
        super().__init__(size=5, show_arrows=True)

class ArrowsRandomS6Env(LavaRandomEnv):
    def __init__(self):
        super().__init__(size=6, show_arrows=True)

class ArrowsRandomS7Env(LavaRandomEnv):
    def __init__(self):
        super().__init__(size=7, show_arrows=True)

register(
    id='MiniGrid-LavaRandomS5-v0',
    entry_point='gym_minigrid.envs:LavaRandomS5Env'
)

register(
    id='MiniGrid-LavaRandomS6-v0',
    entry_point='gym_minigrid.envs:LavaRandomS6Env'
)

register(
    id='MiniGrid-LavaRandomS7-v0',
    entry_point='gym_minigrid.envs:LavaRandomS7Env'
)

register(
    id='MiniGrid-ArrowsRandomS5-v0',
    entry_point='gym_minigrid.envs:ArrowsRandomS5Env'
)

register(
    id='MiniGrid-ArrowsRandomS6-v0',
    entry_point='gym_minigrid.envs:ArrowsRandomS6Env'
)

register(
    id='MiniGrid-ArrowsRandomS7-v0',
    entry_point='gym_minigrid.envs:ArrowsRandomS7Env'
)
