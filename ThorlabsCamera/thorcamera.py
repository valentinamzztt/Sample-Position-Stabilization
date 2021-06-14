"""
Python interface for the Thor scientific Cameras using .Net.

For more information and to get started: https://matham.github.io/thorcam/index.html. 
The Thorlabs custom service told me that the TSI Python SDK does not support the DCC model cameras
"""

from thorcam.camera import ThorCam
class MyThorCam(ThorCam):
    def received_camera_response(self, msg, value):
        super(MyThorCam, self).received_camera_response(msg, value)
        if msg == 'image':
            return
        print('Received "{}" with value "{}"'.format(msg, value))
    def got_image(self, image, count, queued_count, t):
        print('Received image "{}" with time "{}" and counts "{}", "{}"'
              .format(image, t, count, queued_count))
cam = MyThorCam()
cam.start_cam_process()
# get list of attached cams
cam.refresh_cameras()
# open the camera
cam.open_camera('05761')
cam.exposure_range
cam.exposure_ms
# update the exposure value
cam.set_setting('exposure_ms', 150)
cam.exposure_ms
# now play the camera
cam.play_camera()
# now stop playing
cam.stop_playing_camera()
# close the camera
cam.close_camera()
# close the server and everything
cam.stop_cam_process(join=True)
