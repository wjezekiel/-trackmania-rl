# trackmania-rl

WARNING: works for windows or linux only

## Steps to run our pretrained model

1. Follow this tutorial in order to download TrackMania and Openplanet: [https://github.com/trackmania-rl/tmrl/blob/master/readme/get_started.md](https://github.com/trackmania-rl/tmrl/blob/master/readme/Install.md)
2. Open TrackMania2020 and navigate to the settings. Under the video tab, change fullscreen resolution to 1600 x 900.
3. Navigate to the controls tab and scroll to the bottom. Then change camera 3 to use the number 3 button on the keyboard.
4. Create a conda environment with python version 3.10 (conda create -n trackmania python=3.10).
5. Clone this repository and navigate to the directory holding this repository on your computer.
6. Run the following command: pip install -r requirements.txt
7. pip install tmrl
8. Run the following command: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
9. Now navigate to the reward folder in the TMRLdata folder and place the reward.pkl file included in this repository here (For example it was located here on my computer: "C:\Users\sorch\TmrlData\reward\reward.pkl")
10. Now navigate to the folder containing Trackmania2020 and place the map file "Turns.Map.Gbx" in the Maps/My Maps folder (For example this is where my map file was located on my computer: "C:\Users\sorch\Documents\Trackmania2020\Maps\My Maps\Turns.Map.Gbx")
11. Now navigate to the folder containing of this cloned repository and go to its parent folder and paste the save folder in this repository at that location. (For example my save folder was located at "C:\Users\sorch\Documents\GitHub\save" while my github repository is located at "C:\Users\sorch\Documents\GitHub\trackmania-rl")
12. Open TrackMania 2020, click on "Create" on the main menu, then on track editor, and then edit track. Select the track named turns and then click the green flag in the bottom right corner. Press the number 3 button on the keyboard and ensure that the car is not visible in the window, this is very important. If this does not work try pressing the number 0 button on the keyboard. If this also doesn't work refer again to the controls tab in settings and modify until you are able to change to camera 3.
13. Ensure nothing overlaps the window becasue this will interfere with the training process. Now run python train.py, select the TrackMania2020 window, and you should see the car moving by itself after a few seconds. You are safe to deselect the window once is starts running but I would be wary of selecting other windows because we found that it will sometimes interact with them.





