from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import cv2
import os
import json

class SingleAgentCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self,
                 eval_env,
                 verbose=0, 
                 render_frequency=10,
                 deterministic=False,
                 learner='ppo',
                 args={},
                 tile_size=32, 
                 save_every=1000):
        super(SingleAgentCallback, self).__init__(verbose)
        self.iterations_ = 0
        self.render_frequency = render_frequency
        self.eval_env = eval_env
        self.deterministic = deterministic
        world_map = self.eval_env.venv.venv.venv.vec_envs[0].par_env.env.aec_env.env.env.env.ssd_env.world_map
        self.video_resolution = tuple(np.array(world_map.shape[::-1])*tile_size)
        self.args = args
        self.save_every = save_every # save policy every n iterations
        self.learner = learner
        
    def _on_training_start(self) -> None:
        file_name = os.path.join(self.model.logger.dir, "parameters.json")
        os.makedirs(os.path.join(self.logger.dir , "checkpoints"), exist_ok=True)

        json_object = json.dumps(self.args, indent=4)
        with open(file_name, "w") as outfile:
            outfile.write(json_object)

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
            This method will be called by the model after each call to `env.step()`.

            For child callback (of an `EventCallback`), this will be called
            when the event is triggered.

            :return: (bool) If the callback returns False, training is aborted early.
            """
        return True
    
    def _play(self, render=False):
        obs = self.eval_env.reset()
        frames = []
        done = [False] * 2
        rewards = []
        frames = []
        while not (True in done):
            actions, _ = self.model.predict(obs, state=None, deterministic=self.deterministic)
            obs, reward, done, info = self.eval_env.step(actions.astype(np.uint8))
            rewards.append(reward)
            if render:
                frame = self.eval_env.venv.venv.venv.vec_envs[0].par_env.env.aec_env.env.env.env.ssd_env.render(mod="RGB")
                # frames.append(im.fromarray(frame.astype(np.uint8)).resize(size=(720, 480), resample=im.BOX).convert("RGB"))
                frames.append(cv2.resize(frame, self.video_resolution, interpolation=cv2.INTER_NEAREST))
        return np.array(rewards).sum(), frames

    def _on_rollout_end(self) -> None:
        if self.iterations_ % self.render_frequency == 0:
            score, frames = self._play(render=True)
            file_name = self.logger.dir + f"/iteration_{self.iterations_+1}_score_{int(score)}.mp4"
            self.save_video(file_name, frames)
        self.iterations_ += 1
        # if (self.iterations_ % self.save_every) == 0:
        #     self.model.save(os.path.join(self.logger.dir, "checkpoints"))

    def save_video(self, video_path, rgb_arrs, format="mp4v"):
        print("Rendering video...")
        fourcc = cv2.VideoWriter_fourcc(*format)
        video = cv2.VideoWriter(video_path, fourcc, float(15), self.video_resolution)

        for i, image in enumerate(rgb_arrs):
            video.write(image)

        video.release()

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
