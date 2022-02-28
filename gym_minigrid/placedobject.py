from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class PlacedObject(MiniGridEnv):
    '''
    Environment used for rendering translation from arrows to lava
    '''

    def __init__(self, size):
        super().__init__(
            grid_size=size,
            max_steps=4 * size * size,
            # Set this to True for maximum speed
            see_through_walls=True,
            # agent_view_size=3,
        )

    def _gen_grid(self, width, height):
        assert width >= 5 and height >= 5

        # Create an empty grid
        self.grid = Grid(width, height)

        # Place the agent in the top-left corner
        self.agent_pos = (1, 1)
        self.agent_dir = 0

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.goal_pos = np.array((width - 2, height - 2))
        self.put_obj(Goal(), *self.goal_pos)

        self.mission = (
            "avoid the lava and get to the green goal square"
        )

    def set_agent(self, pos, dir):
        self.agent_pos = pos
        self.agent_dir = dir

    def place_lava(self, loc):
        self.put_obj(Lava(), *loc)

    def place_arrow(self, color, orientation, loc):
        self.put_obj(Arrow(color=color, orientation=orientation), *loc)

    def reset(self, agent_pos=None, agent_dir=None):
        super().reset()
        if agent_pos is not None and agent_dir is not None:
            self.set_agent(agent_pos, agent_dir)
        return self.gen_obs()
