**Intersection Insights from Machine Learning**

**Eric Dasmalchi & Andrew Jarnagin**

**UP 229 - Spring 2021**


**Initial Research Questions:** 

* How do motorists, cyclists, and pedestrians interact at an intersection with an unmarked crosswalk? What are basic road user counts at different times and days? Do drivers yield to crossing pedestrians?

* Can a planner with Python skills but no funding or specialized equipment analyze intersection video and obtain accurate, useful data (without relying on a consultant)?

* Can advocates use video analysis as evidence to support the case for safety improvements?


**Context:** 

Cities across the country, including [Los Angeles](http://vision-zero.ua5.land/), have committed to Vision Zero - a policy to completely eliminate traffic-related fatalities. However, despite this lofty goal, fatalities remain stubbornly high. Advocates are pushing local governments, planners, and departments of transportation to make significant investments in safer streets and complete infrastructure for all transportation modes.

![Vision_Zero_chart.JPG](https://github.com/edasmalchi/up229-final-analysis/blob/main/Vision_Zero_chart.JPG)

With granular data on the quantity of pedestrian activity at specific intersections, state and local DOTs could prioritize locations for safety improvements, including marked crosswalks, stop signs, traffic calming infrastructure, bollards, or other interventions. Additionally, community members and safety advocates could provide hard evidence of pedestrian activity levels meeting or exceeding [engineering warrants](http://www.apsguide.org/chapter3_mutcd.cfm) - a highly suspect [methodology](https://trid.trb.org/view.aspx?id=507586) - to bolster the case for changes to the intersection. This data could help municipalities and DOTs target Vision Zero investment.


**Location: The intersection of Motor Ave and Woodbine St in the Palms neighborhood of Los Angeles**

Woodbine is a residential street that meets Motor Ave, which has one lane of general traffic in each direction for cars and bicycles, a center turning lane, and a parking lane on each side. There are stop signs on Woodbine but no traffic control on Motor. There is no marked crosswalk for pedestrians who wish to cross Motor at Woodbine. Nevertheless, there is significant pedestrian crossing at this location.

![Motor_Woodbine.JPG](https://github.com/edasmalchi/up229-final-analysis/blob/main/Motor_Woodbine.JPG)

According to the [California Vehicle Code](https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml?sectionNum=21950.&lawCode=VEH) §21950, *“The driver of a vehicle shall yield the right-of-way to a pedestrian crossing the roadway within any marked crosswalk or within any unmarked crosswalk at an intersection, except as otherwise provided in this chapter.”* At the intersection of Motor Ave and Woodbine St, pedestrians legally have the right-of-way; of course, in practice they usually wait until there are no oncoming vehicles in either direction before beginning to cross.


**Data Collection & Processing**

We recorded video of the intersection from a parked car on Motor Ave at the northeast corner of the intersection (private space on the public right of way!). We collected this data over three separate days. In this process, we learned that cellphones are not ideal for long-term video recording in the hot sun, and a newer phone continually overheated and shut down without saving any data. Luckily, a trusty old Motorola moto g6 saved the day - it could record for about 30 minutes at a time, with a short cooldown period. For future application, a GoPro or similar device built for video capture outdoors is a better option. To reduce the time and processing power needed to analyze our video data, we reduced the quality and lowered the frame rate to 5 frames per second.


**The Tools**

We searched broadly for existing Python libraries and Github repositories that dealt with the combination of machine learning (TensorFlow, Keras, PyTorch, Scikit-learn) and computer vision (OpenCV, CUDA). David Wasserman from Alta Planning + Design also provided some guidance for potential options. This capability does exist off-the-shelf from private companies, but because our intention was to explore methods available to a planner with Python skills, we investigated open-source options.

After reading through several Github repositories and the documentation of various relevant libraries, we settled on the [Object-Detection-and-Tracking repository](https://github.com/yehengchen/Object-Detection-and-Tracking) by Bobby Chen (yehengchen). We opted to work from the [YOLOv3 + SORT](https:/github.com/yehengchen/Object-Detection-and-Tracking/tree/master/OneStage/yolo/yolov3_sort) code, as it seemed sufficient to address our use case. This code pairs YOLO (You Only Look Once) for object detection and SORT to track object across video frames. YOLOv3 is a machine learning model that is already trained and can classify 80 objects (cars and people being the relevant objects for our project); SORT (simple, online, realtime tracking) provides the ability to maintain consistent identification of objects as YOLO identifies them in successive frames. We chose to run the project in Google Colaboratory because it offers GPU processing power that our personal laptops lack. All Colab files, including processed videos, are available *[here](https://drive.google.com/drive/folders/1raq9pOJ4LrKlo3xkqQXnmmYG1sOPfnk3?usp=sharing)*.


**Video Analysis**

The process was not simple! There were several necessary Python libraries with some version conflict issues. More importantly though, the existing code only produced a processed video clip that marked bounding boxes around each object identified, whereas we wanted to obtain the underlying data and return it in a form that we could analyze. With some tinkering with both the main and sort code files, we identified numpy arrays produced at various steps in the detection and tracking process to see how the model analyzed each frame. The array outputs were not intuitive, and it took us some time to figure out what was happening under the hood.

We eventually determined that the code operates as follows:

1) Split the input video file into individual frames

2) Read in a frame of video and detect object classes with a probability score

3) Determine whether the object was detected and tracked in the previous frame

4) Draw bounding boxes on the frame

5) Move to the next frame and repeat the process, separating objects tracked in the prior frame from new detections

6) At the end of the frames, recombine them into a video file. Each frame has bounding boxes drawn around identified objects that are tracked as they move through the video.



**Problems**

* YOLO sometimmes misidentified vehicles and pedestrians, and had trouble dealing consistently with stationary vehicles parked at the curb.
* Our initial hope to train the machine learning model on failure-to-yield events quickly appeared beyond the timeframe and scope of this project. The YOLO model was already trained to detect objects, but failure-to-yield is a much more complicated phenomenon than identifying a common object in a video frame. With more time, we could define failure-to-yield events as follows: any frame in which a pedestrian is located on the curb while a vehicle is moving on the street (over a specified number of frames or time period). We could then set a count of how many frames elapse between the pedestrian's appearance on the curb and their crossing of a centerline in the street. Dividing this count by frames per second would yield the length of pedestrian wait to cross the intersection. This definition is imperfect, but by only including situations where the pedestrian eventually crosses the centerline, we would be able to exclude many irrelevant data points, such as a person waiting at the bus stop or walking north-south along Motor but not crossing.
    * A proof of concept for this approach, including additional commentary on issues, is available in *notebook 2*



**Results**

Our final results did not meet the original project goals (which were admittedly quite high). However, we identified a number of issues, several of which appear possible to resolve with further time. Others would require significantly more data, processing power, and new code.

* The complexity of setting up the necessary runtime environment, compatible libraries, and processing power is likely a barrier to an average planner. Maintaining a working version would require regular testing and updates to ensure each piece works together.

* YOLO classification issues are probably fixable, either by inputting video at a different quality or upgrading to a newer version of the YOLO model.

* Identifying and recording failure-to-yield events would require additional coding, possibly following the defintion described in the Problems section. Possibly, newer machine learning models could trained by manually coding video fragments as failure-to-yield events in order to build a training dataset that could applied to test video. This would likely require significantly more video data and processing power beyond the scope of this project.

* A complete and comprehensive function to analyze intersection video and produce accurate, useful data is likely beyond the capability of an average planner with some Python experience (hence the traditional reliance on third parties for this type of analysis). However, with some more time and resources, a working version is probably possible.
