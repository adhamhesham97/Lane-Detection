import PySimpleGUI as sg
import cv2
import numpy as np
from Lane_Detection import process_frame
from Lane_Detection import ResizeWithAspectRatio
sg.theme("DarkTeal4")

layout = [[sg.T("")], [sg.Text("Choose a video: "), sg.Input(size=(70,30)), sg.FileBrowse(key="-IN-")],[[sg.T("")],sg.Checkbox(text="show details", key="-Details-", default=True),sg.Button("Submit")]]

###Building Window
window = sg.Window('Lane detection', layout, size=(700,150),  element_justification='c')
details = False
  
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event=="Exit":
        break
    
    elif event == "Submit":
        filename = values["-IN-"]
        details = values["-Details-"]
        last_frame=0
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_pos = (10,40)
        text_size = 1.5
        text_color = (0,0,255)
        text_thickness = 4
        
        video = cv2.VideoCapture(filename)
        if (details == False):
            while (video.isOpened()):
                ret, frame = video.read() #boolean indicating success or failure, the captured frame itself
                if ret: #if read was sucessful
                    output = process_frame(frame, details) # detect the lanes
                    cv2.imshow('Lane Detection',output) # show our image to the screen
                    last_frame = output
                    if cv2.waitKey(1) & 0xFF == ord('p'): 
                        cv2.waitKey(0)        
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
            cv2.putText(last_frame,'press any key to exit', (250,520), font, text_size, (255,255,255), text_thickness)
            cv2.imshow('Lane Detection',last_frame)
            cv2.waitKey(0) #waitkey(0) means wait until a key is pressed
            video.release()
            cv2.destroyAllWindows()
        else:
            while (video.isOpened()):
                ret, frame = video.read() #boolean indicating success or failure, the captured frame itself
                if ret: #if read was sucessful
                    output = process_frame(frame, details) # detect the lanes
                    
                    # add images titles
                    cv2.putText(output[0],'Greyscale',text_pos, font, text_size, text_color, text_thickness)
                    cv2.putText(output[1],'Canny Edges',text_pos, font, text_size, text_color, text_thickness)
                    cv2.putText(output[2],'Filtered Edges',text_pos, font, text_size, text_color, text_thickness)
                    cv2.putText(output[3],'Lane Edges',text_pos, font, text_size, text_color, text_thickness)
                    cv2.putText(output[4],'Final Output',text_pos, font, 1, text_color, 2)
                    
                    # decrease resolution of the details to 1/4
                    for i in range(4):
                        output[i]=cv2.resize(output[i], (0, 0), None, 0.25, 0.25)
                        
                    x1 = np.vstack(output[0:4]) # concatenate the images horizontaly
                    output = np.hstack((x1, output[4])) # concatenate the images with the oputput verticaly
                    output = ResizeWithAspectRatio(output, width=1280) # resize to fit window        
                    cv2.imshow('Lane Detection',output) #show our image to the screen
                    last_frame = output
                    if cv2.waitKey(1) & 0xFF == ord('p'): 
                        cv2.waitKey(0)        
                    elif cv2.waitKey(1) & 0xFF == ord('q'): 
                        break
                else:
                    break
            cv2.putText(last_frame,'press any key to exit', (550,520), font, text_size, (255,255,255), text_thickness)
            cv2.imshow('Lane Detection',last_frame)
            cv2.waitKey(0) #waitkey(0) means wait until a key is pressed
            video.release()
            cv2.destroyAllWindows()
