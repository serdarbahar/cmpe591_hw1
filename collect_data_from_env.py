import numpy as np
from homework1 import Hw1Env

env = Hw1Env(render_mode="offscreen")
num_data = 1000
images = np.zeros((num_data, 3, 128, 128))
actions = np.zeros((num_data, 1))
positions_after = np.zeros((num_data, 2))
images_after = np.zeros((num_data, 3, 128, 128))

for i in range(num_data):
    env.reset()
    action_id = np.random.randint(4)
    pos_before, img_before = env.state()
    env.step(action_id)
    pos_after, img_after = env.state()

    images[i] = np.array(img_before)
    actions[i] = action_id
    positions_after[i] = pos_after
    images_after[i] = np.array(img_after)
    env.reset()
    print(f"{i} samples collected")

np.save("data/images", images)
np.save("data/actions", actions)
np.save("data/positions_after", positions_after)
np.save("data/images_after", images_after)




