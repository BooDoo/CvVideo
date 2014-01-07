import os, sys, subprocess
import cv2
from fractions import Fraction

class CvVideo(object):
  #Constructor for the CvVideo object (requires OpenCV2 with ffmpeg support)
  def __init__(self, input_file, gif_path='gifs', temp_path='tmp', splitter='___', crop_factor=None, crop_width=None, crop_height=None, from_youtube=None, avi_codec=None, avi_fps=None):
    '''
    crop_factor as float applies against both width and height, otherwise pass tuple (w_factor, h_factor)
    '''
    self.input_file = input_file
    self.input_file_tail = os.path.split(input_file)[1]

    self.stream = stream = cv2.VideoCapture(input_file)
    self.framecount = stream.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    self.fps = stream.get(cv2.cv.CV_CAP_PROP_FPS)
    self.duration = self.framecount / self.fps
    self.width = stream.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    self.height = stream.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)   
    self.fourcc = stream.get(cv2.cv.CV_CAP_PROP_FOURCC)

    if from_youtube:
      try:
        self.uploader, self.vid_id = os.path.splitext(self.input_file_tail)[0].split(splitter)[0:2]
      except ValueError as e:
        self.uploader, self.vid_id = 'Unknown', os.path.splitext(self.input_file_tail)[0]
      self.vid_link = "http://youtube.com/watch?v=" + self.vid_id

    self.aspect_ratio = Fraction(int(self.width),int(self.height)).limit_denominator(10)
    self.template_scale = round(self.width/640.0*100)/100
    self.img = None
    self.gray = None
    self.template_found = None
    self.templates = None

    try:
      out_base = self.vid_id
    except NameError:
      out_base = self.input_file_tail
    self.out_gif = os.path.join(gif_path, out_base + '.gif')
    self.out_avi = os.path.join(temp_path, out_base + '.avi')
    self.out_mp4 = os.path.join(temp_path, out_base + '.mp4')

    if crop_factor:
      if type(crop_factor) == float:
        self.crop_width, self.crop_height = (int(min(self.width*crop_factor, self.width)), int(min(self.height*crop_factor, self.height)) )
      else:
        self.crop_width, self.crop_height = (int(self.width*crop_factor[0]), int(self.height*crop_factor[1]))
    elif crop_width or crop_height:
      self.crop_width, self.crop_height = (int(min(crop_width or sys.maxint, self.width)), int(min(crop_height or sys.maxint, self.height)) )

    self.avi_codec = avi_codec or 0
    self.avi_fps = avi_fps or self.fps
    self.output = cv2.VideoWriter(self.out_avi, self.avi_codec, self.avi_fps, (self.crop_width,self.crop_height))
    self.roi_ratio = Fraction(int(self.crop_width),int(self.crop_height)).limit_denominator(10)
    self.roi_reset()

  def __getitem__(self, key):
    return getattr(self, key)

  def __setitem__(self, key, value):
    setattr(self, key, value)
    return value

  @property
  def roi_default(self):
    """return our default crop ROI"""
    self._minY, self._minX = [int( (self.height - self.crop_height) / 2), int( (self.width - self.crop_width) / 2)]
    self._maxY, self._maxX = [self._minY + self.crop_height, self._minX + self.crop_width]
    return (self._minX, self._minY, self._maxX, self._maxY)

  @property
  def frame(self):
    """A pass-through to VideoCapture.POS_FRAMES"""
    return self.stream.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)

  @frame.setter
  def frame(self, frame):
    #print "Setting 'frame' to",frame,"out of",self.framecount
    if frame < 0 or frame > self.framecount:
      raise cv2.error("Requested frame is out of bounds")
    else:  
      self.stream.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame)

  @property
  def time(self):
    """Current position in the video (in seconds)"""
    return self.frame / self.fps
    
  @time.setter
  def time(self, seconds):
    #self.stream.set(cv2.cv.CV_CAP_PROP_POS_MSEC, seconds * 1000.0)
    self.frame = int(self.fps * seconds)
  
  @property
  def roi_rect(self):
    """The frame we use for cropping (ROI)"""
    return self._roi_rect
  
  @roi_rect.setter
  def roi_rect(self, rect):
    minX, minY, maxX, maxY = rect
    if minX < 0 or minY < 0 or maxX < 0 or maxY < 0 or maxX < minX or maxY < minY or maxX > self.width or maxY > self.height:
      raise cv2.error("Invalid dimensions for crop/ROI rectangle.")
    else:
      self._roi_rect = (minX, minY, maxX, maxY)

  @property
  def roi(self):
    """Subset of pixels (ROI) from current frame"""
    if self.img == None:
      self.read()
    return self.img[self.roi_rect[1]:self.roi_rect[3], self.roi_rect[0]:self.roi_rect[2]]
  
  @roi.setter
  def roi(self, rect):
    self.roi_rect = rect

  @property
  def roi_gray(self):
    """Subset of pixels (ROI) from current frame, in grayscale"""
    if self.gray == None:
      self.read()
    return self.gray[self.roi_rect[1]:self.roi_rect[3], self.roi_rect[0]:self.roi_rect[2]]

  def roi_reset(self):
    """Reset roi_rect to default for GIF output"""
    self.roi_rect = self.roi_default
    return self #chainable

  def get_sum(self, color=False, use_roi=False, roi_rect=None):
    if self.gray == None:
      self.read()

    if use_roi:
      frame = get_roi(color=color, roi_rect=roi_rect)
    else:
      frame = self.gray

    return frame.sum()

  @property
  def nonzero_count(self, color=False, use_roi=False, roi_rect=None):
    if self.gray == None:
      self.read()

    if use_roi:
      frame = get_roi(color=color, roi_rect=roi_rect)
    else:
      frame = self.gray

    return len(frame.nonzero()[0])

  @property
  def nonzero_factor(self, color=False, use_roi=False, roi_rect=None):
    if self.gray == None:
      self.read()

    if use_roi:
      frame = get_roi(color=color, roi_rect=roi_rect)
    else:
      frame = self.gray

    return len(frame.nonzero()[0]) / float(frame.size)

  def set_frame(self, frame=0):
    """A chainable alias for CvVideo.frame = `frame`"""
    self.frame = frame
    return self #chainable
  
  def get_roi(self, color=True, roi_rect=None):
    """Function to get roi with custom params"""
    if self.img == None:
      self.read()

    if not roi_rect:
      return self.roi if color else self.roi_gray
    else:
      source = self.img if color else self.gray
      roi_rect = roi_rect if roi_rect else self.roi_default
      return source[roi_rect[1]:roi_rect[3], roi_rect[0]:roi_rect[2]]
  
  def _skip(self, frames=1):
    """Generic function for scrubbing back/forward by `frames`"""
    #print "Starting at",self.frame,"skipping",frames,"frames"
    self.frame += frames
    sys.stdout.write("+" if frames>0 else "-")
    sys.stdout.flush()
    return self #for chaining
    
  def read(self, frame=None):
    """Read next frame from stream and save image to img/gray properties"""
    if frame:
      self.frame = frame
    ret, self.img = self.stream.read()
    self.gray = cv2.cvtColor(self.img, cv2.cv.CV_BGR2GRAY)
    return self #for chaining
    
  def read_frame(self, frame=None):
    """Convenience alias for `read()`"""
    return self.read(frame) #chainable

  def read_time(self, seconds):
    """Go to `seconds` in stream, read image to img/gray properties""" 
    frame = self.fps*seconds
    return self.read_frame(frame) #chainable

  def skip_frames(self, frames=1):
    """Public alias for `_skip(frames)`"""
    return self._skip(frames).read() #chainable

  def skip_time(self, seconds=1):
    """Turn `seconds` into frames and then `_skip` by that amount"""
    return self.skip_frames(self.fps*seconds) #chainable
    
  def skip_forward(self, seconds=1):
    """Convenience alias for `skip_frames`"""
    return self.skip_frames(self.fps*seconds) #chainable
    
  def skip_back(self, seconds=5):
    """Convenience function goes backwards by `seconds`"""
    return self.skip_frames(self.fps * seconds * -1) #chainable

  def frame_to_file(self, out_file='frame.png', color=True, frame=-1, use_roi=False, roi_rect=None):
    """Write current frame (or specified `frame`) to `out_file`, optionally in grayscale and/or cropped to `roi_rect`"""
    if use_roi and not roi_rect:
      roi_rect = self.roi_default
    
    if frame >= 0: #Target specific frame?
      self.frame = frame
    self.read()
    
    if use_roi and not color:
      cv2.imwrite(out_file, self.get_roi(False, roi_rect) )
    elif use_roi and color:
      cv2.imwrite(out_file, self.get_roi(True, roi_rect) )
    elif not color:
      cv2.imwrite(out_file, self.gray)
    else:
      cv2.imwrite(out_file, self.img)
    
    return self #chainable
  
  def frame_to_output(self, color=True, frame=-1, use_roi=False, roi_rect=None):
    """Write current frame (or specified `frame`) to `CvVideo.output` buffer, optionally in grayscale and/or cropped to `roi_rect`"""
    if not self.output:
      raise cv2.error("No output stream for writing!")
    
    if use_roi and not roi_rect:
      roi_rect = self.roi_default
    
    if frame >= 0: #Target specific frame?
      self.frame = frame
    self.read()
    
    #dump output frames
    #cv2.imwrite('dump/'+ self.out_base + str(int(frame)) + '.png', self.roi)

    if use_roi and not color:
      self.output.write(self.get_roi(False, roi_rect) )
    elif use_roi and color:
      self.output.write(self.get_roi(True, roi_rect) )
    elif not color:
      self.output.write(self.gray)
    else:
      self.output.write(self.img)
  
    return self #chainable
  
  def reset_output(self, codec=None, fps=None):
    codec = codec or self.avi_codec or 0
    fps = fps or self.avi_fps or self.fps
    self.output = cv2.VideoWriter(self.out_avi, codec, fps, (self.crop_width,self.crop_height))
    return self #chainable

  #interval is in seconds, can be negative.
  def clip_to_output(self, from_frame=-1, to_frame=-1, frame_skip=None, interval=None, duration=None, color=True, use_roi=False, roi_rect=None):
    """Take a clip of input `stream` and write to `output` buffer as series of frames"""
    if use_roi and not roi_rect:
      roi_rect = self.roi_rect
    
    if from_frame < 0: #no specific frame? start from where we are
        from_frame = self.frame
    
    if not frame_skip: #use frame_skip if specified
      frame_skip = interval*self.fps if interval else 1 #frame-by-frame default
    
    if to_frame >= 0: #use to_frame if specified
      duration = None
    elif duration: #otherwise use duration if given
      to_frame = from_frame + duration*self.fps
    else: #still no value? use last frame
      to_frame = self.framecount-1
    
    if to_frame < from_frame: #make sure our loop isn't infinite!
      frame_skip = abs(frame_skip) * -1
    else:
      frame_skip = abs(frame_skip)
    
    #set a timestamp where we start:
    self.clip_start = self.time
    
    #ensure ints
    from_frame = int(from_frame)
    to_frame = int(to_frame)
    frame_skip = int(frame_skip)
    
    #do it
    for frame in range(from_frame, to_frame, frame_skip):
      self.frame_to_output(color, frame, use_roi, roi_rect)
      
    return self #chainable
  
  def clip_to_mp4(self, out_file=None, from_frame=-1, to_frame=-1, duration=None, color=True, transpose=None, use_roi=False, roi_rect=None, filters=None, scale=None, crop=None, scale_width=None, scale_height=None):
    '''Need to support a lot more control here! Right now it's very specific to making a SnapChat video.'''
    if use_roi and not roi_rect:
      roi_rect = self.roi_rect

    if not out_file:
      out_file = self.out_mp4

    if from_frame < 0:
      from_frame = self.frame
    if to_frame < 0:
      if not duration:
        to_frame = from_frame + 4 * self.fps
      else:
        to_frame = from_frame + duration * self.fps

    if not filters:
      filters = []
    elif type(filters) == list:
      filters = ",".join(filters)

    if not color:
      filters.append("colorchannelmixer=.3:.4:.3:0:.3:.4:.3:0:.3:.4:.3")

    if crop:
      filters.append("crop=%s"%crop)
    elif use_roi:
      crop_w, crop_h, crop_x, crop_y = (roi_rect[2] - roi_rect[0], roi_rect[3] - roi_rect[1], roi_rect[0], roi_rect[1])
      filters.append("crop=%s:%s:%s:%s"%(crop_w, crop_h, crop_x, crop_y) )

    if scale_width:
      filters.append("scale=w=%i:h=-1"%scale_width)
    elif scale_height:
      filters.append("scale=h=%i:w=-1"%scale_height)
    elif scale:
      filters.append("scale=in_w*"+str(scale)+":-1")

    if transpose:
      filters.append("transpose="+str(transpose))

    sys.stdout.write("\nMaking MP4 at:"+out_file+"...")
    sys.stdout.flush()

    command = ['ffmpeg', '-y']
    command.extend(['-ss', str(from_frame / self.fps)])
    command.extend(['-i', self.input_file])
    if duration:
      command.extend(['-t', str(duration)])
    else:
      command.extend(['-to', str(from_frame / self.fps)])
    command.extend(['-vf', ",".join(filters)])
    command.extend(['-vcodec', 'libx264'])
    command.extend(['-vprofile', 'baseline'])
    command.extend(['-acodec', 'copy'])
    command.extend(['-movflags', 'faststart'])
    command.append(out_file)
    subprocess.call(command)

    sys.stdout.write("done!\n")
    sys.stdout.flush()

    return self #chainable

  def gif_from_out_avi(self, out_file=None, color=False, brightness=100, saturation=100, hue=100, delay=10, fuzz="4%", layers="OptimizeTransparency", flush_map=True):
    """Call ImageMagick's `convert` from shell to create a GIF of video file found at `out_avi`"""
    #clear tmp folder:
    #subprocess.call(['sudo', 'rm', '-r', '/tmp'])

    if not out_file:
      out_file = self.out_gif

    if not color:
      saturation = 0

    #values associated with `-modulate`
    bsh = map(str, [brightness, saturation, hue])
    
    try:
      if os.path.getsize(self.out_avi) < 6000:
        raise cv2.error("Didn't write any frames to AVI. Wrong crop-size? Wrong codec?")
    except os.error as e:
      raise cv2.error("Temp AVI doesn't exist!")

    sys.stdout.write("\nWriting to "+out_file+"...")
    sys.stdout.flush()
    
    command = ['convert']
    if delay > 0:
      command.extend(['-delay', str(delay)])

    command.append(self.out_avi)

    if not all([v == '100' for v in bsh]):
      command.extend(['-modulate', ",".join(bsh)])
    if fuzz:
      command.extend(['-fuzz', str(fuzz)])
    if layers:
      command.extend(['-layers', str(layers)])
    if flush_map:
      command.extend(['+map'])
    command.append(out_file)

    subprocess.call(command)

    sys.stdout.write("done!\n")
    sys.stdout.flush()
    return self #chainable

  def clear_out_avi(self):
    """Delete the file at location `out_avi`"""
    try:
      os.remove(self.out_avi)
    except IOError as e:
      print e

    return self #chainable

  #template methods are NOT chainable!
  def template_check(self, templates=None, threshold=0.84, method=cv2.TM_CCOEFF_NORMED, use_roi=False, roi_rect=None):
    """Cycle through each image in `templates` and perform OpenCV `matchTemplate` until match found (or return False)"""
    #TODO: Enable checking against min_val for methods where that's more appropriate
    
    if templates == None and hasattr(self, 'templates'):
      templates = self.templates
    elif templates == None:
      raise cv2.error("No template(s) to match against!")
    
    roi_rect = roi_rect if roi_rect else self.roi_default
    
    target = self.get_roi(False, roi_rect) if use_roi else self.gray
    
    #dump checked frames
    #cv2.imwrite('dump/'+ self.out_base + '/' + str(int(self.frame)) + '.png', target)
    
    for label,template in templates:
      res = cv2.matchTemplate(target, template, method)
      min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
      if max_val >= threshold:
        #cv2.imwrite('dump/'+ self.out_base + '/' + str(int(self.frame)) + '-found.png', target)
        self.template_found = label
        #print "max_val for %s was %f" % (label, max_val)
        return True
    
    return False

  def template_best(self, templates=None, threshold=0.84, method=cv2.TM_CCOEFF_NORMED, use_roi=False, roi_rect=None):
    """Cycle through each image in `templates` and perform OpenCV `matchTemplate`, return best match (or return False)"""
    #TODO: Enable checking against min_val for methods where that's more appropriate
    
    if templates == None and hasattr(self, 'templates'):
      templates = self.templates
    elif templates == None:
      raise cv2.error("No template(s) to match against!")
    
    roi_rect = roi_rect if roi_rect else self.roi_default
    
    target = self.get_roi(False, roi_rect) if use_roi else self.gray
    
    #dump checked frames
    #cv2.imwrite('dump/'+ self.out_base + '/' + str(int(self.frame)) + '.png', target)
    
    matches = {} #dict((label,None) for label,template in templates)
    
    for label,template in templates:
      res = cv2.matchTemplate(target, template, method)
      min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
      if max_val >= threshold:
        matches[label] = max_val
    
    if matches:
      match_best = [label for label in sorted(matches, key=matches.get, reverse=True)][0]
      self.template_found = match_best
      return match_best
    
    return False
  
  def until_template(self, interval=0.5, templates=None, threshold=0.84, method=cv2.TM_CCOEFF_NORMED, frame_skip=None, use_roi=False, roi_rect=None, max_length=None):
    """Scrub through video until a template is found"""
    frame_skip = frame_skip if frame_skip else interval*self.fps
    max_length = max_length if max_length else self.duration
    
    first_frame = self.frame
    max_length *= self.fps
    
    while abs(self.frame - first_frame) < max_length and not self.template_check(templates, threshold, method):
      self.skip_frames(frame_skip)
    
    if abs(self.frame - first_frame) > max_length:
      return False
    else:
      return True

  def while_template(self, interval=0.5, templates=None, threshold=0.84, method=cv2.TM_CCOEFF_NORMED, frame_skip=None, use_roi=False, roi_rect=None, max_length=None):
    """Scrub through video until no template is matched"""
    frame_skip = frame_skip if frame_skip else interval*self.fps
    max_length = max_length if max_length else self.duration
    
    first_frame = self.frame
    max_length *= self.fps
    
    while abs(self.frame - first_frame) < max_length and self.template_check(templates, threshold, method):
      self.skip_frames(frame_skip)
    
    if abs(self.frame - first_frame) > max_length:
      return False
    else:
      return True
