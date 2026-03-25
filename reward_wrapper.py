import numpy as np

class RewardWrapper:
    def __init__(self, env, w_far=2, w_near=4, w_push=3, w_ir=8, w_front=2):
        self.env = env
        self.w_far = w_far
        self.w_near = w_near
        self.w_push = w_push
        self.w_ir = w_ir
        self.w_front = w_front

    def reset(self, **kwargs):
        s = self.env.reset(**kwargs)
        self.prev_push = False
        self.stuck_rotation_count = 0
        return s

    def step(self, action, render=False):
        s2, r_env, done = self.env.step(action, render=render)
        r = r_env

        # ===== SENSOR GROUPING =====

        # RIGHT
        near_right = s2[[0, 2]]
        far_right  = s2[[1, 3]]

        # FRONT
        near_front = s2[[4, 6, 8, 10]]
        far_front  = s2[[5, 7, 9, 11]]

        # LEFT
        near_left = s2[[12, 14]]
        far_left  = s2[[13, 15]]

        # IR & STUCK
        ir = s2[16]
        stuck = s2[17]

        # ===== SENSOR COUNTS =====

        far_active = np.sum(far_right) + np.sum(far_front) + np.sum(far_left)
        near_active = np.sum(near_right) + np.sum(near_front) + np.sum(near_left)

        # ===== BASE SHAPING =====

        r += self.w_far * far_active
        r += self.w_near * near_active
        r += self.w_ir * ir

        # forward alignment bias
        front_bias = np.sum(near_front) + 0.5 * np.sum(far_front)
        r += self.w_front * front_bias

        # attach bonus
        if (not self.prev_push) and self.env.enable_push:
            r += 300

        # pushing reward
        if self.env.enable_push:
            r += self.w_push

        # no-signal penalty
        if near_active == 0 and far_active == 0 and not self.env.enable_push:
            r -= 2.0

        # mild stuck penalty (env already gives big one)
        if stuck:
            r -= 10.0

        # ===== CONTROL LEARNING (KEY PART) =====

        # ---- Forward primitive ----
        if action == "FW":
            if not stuck:
                r += 3.0
            else:
                r -= 5.0

        # ---- Bounded rotation recovery ----
        if stuck:
            if action in ["L45", "R45", "L22", "R22"]:
                self.stuck_rotation_count += 1

                if self.stuck_rotation_count <= 3:
                    if action in ["L45", "R45"]:
                        r += 6.0
                    else:
                        r += 3.0
                else:
                    r -= 4.0

            elif action == "FW":
                self.stuck_rotation_count = 0

        else:
            self.stuck_rotation_count = 0

        # ---- Clip extreme negatives (stability) ----
        if r < -150:
            r = -150

        self.prev_push = self.env.enable_push
        return s2, float(r), done