# Variables for inference.py
# Don't change if not necessary

GRID = 8
size = (128, 128)
BLUR_TYPE = "HARD_BLUR" # "LIGHT_BLUR" is available
PADDINGTON = size[0] / GRID / GRID

# Main ffmpeg command. Change it for different quality / speed.
def ffmpeg_command(video_shape, FPS, OUTPUT_VIDEO):
	command = [
	    'ffmpeg',
	    '-y',
	    '-f', 'rawvideo',
	    '-vcodec', 'rawvideo',
	    '-s', f'{video_shape[0]}x{video_shape[1]}',
	    '-pix_fmt', 'bgr24',
	    '-r', str(FPS),
	    '-i', '-',
	    '-vcodec', 'libx264',
	    '-preset', 'veryfast',
	    '-crf', '28',
	    '-pix_fmt', 'yuv420p',
	    '-tune', 'zerolatency', 
	    OUTPUT_VIDEO
	]

	return command