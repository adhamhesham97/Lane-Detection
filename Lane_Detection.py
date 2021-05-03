import numpy as np
import matplotlib.pyplot as plt
import cv2

def process_frame (frame, details=False): 
    
    # convert image (frame) to greyscale
    grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    
    # remove noise using low pass filter
    filter_size = 5
    blur_image = cv2.GaussianBlur(grey_image,(filter_size, filter_size),0) 
    
    # detect edges using canny  
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_image, low_threshold, high_threshold) #cv2.Canny returns an image
    
    # filter the edges that concern us only (lanes)
    # lanes will exist in a trapeziod area in the bottom of the frame
    mask = np.zeros_like(edges)
    imshape = frame.shape
    vertices = np.array([[(0,imshape[0]),(imshape[1]/2-25,imshape[0]*0.6),(imshape[1]/2+25,imshape[0]*0.6), (imshape[1],imshape[0])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)    
    filtered_edges = cv2.bitwise_and(edges, mask)
    
    # detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(filtered_edges, rho = 2, theta = np.pi/180, threshold = 15,
                                minLineLength = 40, maxLineGap = 30)
    
    # draw detected lines on the frame
    lane_color = (255,100,0)
    lanes_image = np.zeros_like(frame) 
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(lanes_image,(x1,y1),(x2,y2),lane_color,5)
    output_image = cv2.addWeighted(frame, 0.8, lanes_image, 1, 0)
    
    if(details==False): return output_image
    else: return [greyToRGB(grey_image), greyToRGB(edges), greyToRGB(filtered_edges), lanes_image, output_image]

def greyToRGB(grey):
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


#%% one frame

# frame = cv2.imread('test_images/image1.jpg')
# output = process_frame(frame)
# output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
# plt.imshow(output)


#%% video

# video = cv2.VideoCapture('test_videos/video1.mp4')
# details = True
# if (details == False):
#     while (video.isOpened()):
#         ret, frame = video.read() #boolean indicating success or failure, the captured frame itself
#         if ret: #if read was sucessful
#             output = process_frame(frame, details) # detect the lanes
#             cv2.imshow('Lane Detection - Frame',output) # show our image to the screen
#             if cv2.waitKey(1) & 0xFF == ord('p'): 
#                 cv2.waitKey(0)        
#             elif cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         else:
#             break
#     cv2.waitKey(0) #waitkey(0) means wait until a key is pressed
#     video.release()
#     cv2.destroyAllWindows()
# else:
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     text_pos = (10,40)
#     text_size = 1.5
#     text_color = (0,0,255)
#     text_thickness = 4
#     while (video.isOpened()):
#         ret, frame = video.read() #boolean indicating success or failure, the captured frame itself
#         if ret: #if read was sucessful
#             output = process_frame(frame, details) # detect the lanes
            
#             # add images titles
#             cv2.putText(output[0],'Greyscale',text_pos, font, text_size, text_color, text_thickness)
#             cv2.putText(output[1],'Canny Edges',text_pos, font, text_size, text_color, text_thickness)
#             cv2.putText(output[2],'Filtered Edges',text_pos, font, text_size, text_color, text_thickness)
#             cv2.putText(output[3],'Lane Edges',text_pos, font, text_size, text_color, text_thickness)
#             cv2.putText(output[4],'Final Output',text_pos, font, 1, text_color, 2)
            
#             # decrease resolution of the details to 1/4
#             for i in range(4):
#                 output[i]=cv2.resize(output[i], (0, 0), None, 0.25, 0.25)
                
#             x1 = np.vstack(output[0:4]) # concatenate the images horizontaly
#             output = np.hstack((x1, output[4])) # concatenate the images with the oputput verticaly
#             output = ResizeWithAspectRatio(output, width=1280) # resize to fit window        
#             cv2.imshow('Lane Detection - Frame',output) #show our image to the screen
#             if cv2.waitKey(1) & 0xFF == ord('p'): 
#                 cv2.waitKey(0)        
#             elif cv2.waitKey(1) & 0xFF == ord('q'): 
#                 break
#         else:
#             break
#     cv2.waitKey(0) #waitkey(0) means wait until a key is pressed
#     video.release()
#    cv2.destroyAllWindows()

