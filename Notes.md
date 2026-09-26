>> This File contains the notes and the important details of the topics covered in the codes:
Some prerequsites:(a). OpenCV takes the deafault color space as BGR(Blue,Green,Red).
                  (b). Grey images are single color channel images
1. Interpolation in resizing images :
-> In image resizing, interpolation is the mathematical method used to calculate the color and intensity of new pixels when an image is made larger or smaller.
 When you change an image's size, you are changing the number of pixels in its grid. The computer has to figure out what values to assign to the new grid spaces:
                (a):When scaling up (Zooming in): The image expands, leaving blank spaces between the original pixels. Interpolation guesses and fills in the missing colors.
                (b):When scaling down (Zooming out): Pixels must be squished together. Interpolation blends and averages multiple pixels into one.
example: 1.cv.INTER_LINEAR-when resizing an image to a larger image you will usually use this function
         2.cv.INTER_CUBIC-it is also used whhen resizing to a larger value ,it is slower than linear or area method but it gives a much clearer image

2. warpAffine():cv2.warpAffine() is the worker engine that actually applies geometric transformations to an image.
                Functions like cv.getRotationMatrix2D() or your manual transMat only calculate the instructions (the matrix). warpAffine() takes those instructions and physically moves the pixels to create the new image.

3. Contours:In everyday language, a contour is the outline of a form.
           In digital image processing and computer vision (such as when using tools like OpenCV), it has a specific technical definition: 
                   A contour is a line that joins all continuous points along a boundary that share the same color or intensity.
        >>Difference between edges and contours:
         >Edges are local areas in an image where the brightness changes sharply. They are just raw pixels where contrast shifts.
         >Contours are the bigger picture. They take those edge pixels and connect them together into a meaningful, continuous boundary line that forms a complete shape

4. 