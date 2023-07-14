import imageio_ffmpeg
import pybullet as p

def record_video(env, model):
    cam_width, cam_height = 480, 360
    vid = imageio_ffmpeg.write_frames('vid.mp4', (cam_width, cam_height), fps=30)
    vid.send(None)  # seed the video writer with a blank frame

    for i in range(1):
        obs, _ = env.reset()
        for j in range(2000):
            action, _states = model.predict(obs)
            obs, rewards, dones, info, _ = env.step(action)
            if j % 8 == 0: # pybullet steps at 240fps, but we want 30fps for video
                image = env.render()
                vid.send(image)

    vid.close()
    p.disconnect()