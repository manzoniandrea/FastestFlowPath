

import matplotlib.pyplot as plt
import math
import numpy as np

import os
import time
import subprocess
import glob
import pandas as pd

def distance(p1, p2):
    """Calculates the Euclidean distance between two points."""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def sort_points_by_distance(points, p1):
    """Sorts a list of points by their distance from point p1."""
    return sorted(points, key=lambda p2: distance(p1, p2))

def find_intersecting_points(p1, p2, grid_extents, resolution):
    # Unpack the segment points
    x1, y1 = p1
    x2, y2 = p2

    # Unpack the grid extents
    x_min, x_max, y_min, y_max = grid_extents

    # Unpack the resolution
    del_R, del_C = resolution

    # Calculate the slope and y-intercept of the segment
    if (x2 - x1) == 0:
        # Determine the range of y-values within the grid extents
        y_start = max(y_min, min(y1, y2))
        y_end = min(y_max, max(y1, y2))

        # Find the intersection points with the grid lines
        intersecting_points = []
        for y in range(int(y_start / del_C), int(y_end / del_C) + 1):
            x = x1

            if y_min <= y* del_C <= y_max:    
                xpt = x 
                if min([x1,x2]) <= xpt <= max([x1,x2]) and  min([y1,y2]) <= y* del_C <= max([y1,y2]):
                    intersecting_points.append((x, y*del_C))
        
    else:
        slope = (y2 - y1) / (x2 - x1)
        y_intercept = y1 - slope * x1

        # Determine the range of x-values within the grid extents
        x_start = max(x_min, min(x1, x2))
        x_end = min(x_max, max(x1, x2))

        # Find the intersection points with the grid lines
        intersecting_points = []
        for x in range(int(x_start / del_R), int(x_end / del_R) + 1):
            y = slope * x * del_R + y_intercept
            if y_min <= y <= y_max:    
                xpt = x * del_R
                if min([x1,x2]) <= xpt <= max([x1,x2]) and  min([y1,y2]) <= y <= max([y1,y2]):
                    intersecting_points.append((x * del_R, y))

        # Determine the range of y-values within the grid extents
        y_start = max(y_min, min(y1, y2))
        y_end = min(y_max, max(y1, y2))

        # Find the intersection points with the grid lines
        #intersecting_points = []
        for y in range(int(y_start / del_C), int(y_end / del_C) + 1):
            if slope == 0:
                x = max([x1,x2])+1                
            else:
                x = (y * del_C - y_intercept)/slope
            if y_min <= y* del_C <= y_max:    
                xpt = x 
                if min([x1,x2]) <= xpt <= max([x1,x2]) and  min([y1,y2]) <= y* del_C <= max([y1,y2]):
                    intersecting_points.append((x, y*del_C))
        
    sorted_points = sort_points_by_distance(intersecting_points, p1)
    # Plot the line, grid, and intersecting points
#     plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', label='Line')
#     plt.grid(True, linestyle='-')
#     plt.xticks(np.arange(x_min, x_max + 1, del_R))
#     plt.yticks(np.arange(y_min, y_max + 1, del_C))
#     if len(intersecting_points) > 0:
#         plt.scatter(*zip(*sorted_points), color='red', label='Intersecting Points')
#     plt.legend()
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.title('Line and Grid')
#     plt.show()

    return sorted_points

def calculate_intermediate_points(points):
    intermediate_points = []
    for i in range(len(points)-1):
        x1, y1 = points[i]
        x2, y2 = points[i+1]
        intermediate_x = (x1 + x2) / 2
        intermediate_y = (y1 + y2) / 2
        intermediate_points.append((intermediate_x, intermediate_y))
    return intermediate_points

import numpy as np
from skimage.draw import line


def sample_grid_points(extent, y_field, points):
    # Extract grid extent dimensions
    xmin, xmax, ymin, ymax = extent

    # Get the size of the grid
    grid_width = y_field.shape[1]
    grid_height = y_field.shape[0]

    # Initialize the list to store the sampled values
    sampled_values = []

    # Iterate over the points and sample the grid values
    for point in points:
        # Extract the x and y coordinates of the point
        x, y = point

        # Convert the coordinates to grid indices
        col_index = int((x - xmin) / (xmax - xmin) * grid_width)
        row_index = int((y - ymin) / (ymax - ymin) * grid_height)

        # Check if the grid indices are within bounds
        if 0 <= row_index < grid_height and 0 <= col_index < grid_width:
            # Sample the value from the grid
            sampled_value = y_field[row_index, col_index]
            sampled_values.append(sampled_value)
        else:
            # Handle points outside the grid extent
            sampled_values.append(None)

    return sampled_values

def remove_consecutive_duplicates(lst):
    result = []
    prev_value = None
    for value1 in lst:
        if value1 != prev_value:
            result.append(value1)
        prev_value = value1
    return result


def check_file(folder, value, xPlane):
    # Get the list of files in the folder sorted by modification time
    files = glob.glob(os.path.join(folder, "*"))
    files.sort(key=os.path.getmtime)
    
    x = []
    y = []
    z = []  

    if files:
        # Get the last modified file
        last_file = files[-1]
        #print(last_file)
        # Read the file using pandas
        df = pd.read_csv(last_file)
        iterationStr = 'p-'+ str(value)
        if 'snap' in last_file and iterationStr in last_file:
            if df['x coord'].max() > xPlane:
                filtered_df = df[df["x coord"] > xPlane]
                max_row = filtered_df[filtered_df["x coord"] == filtered_df["x coord"].max()]
                idpt = filtered_df['id'].values[0]
                
                # Iterate over the file list
                for file in files:
                    nameIter = 'snap-' + str(value)
                    if nameIter in file:
                        # Create a DataFrame from the file
                        df1 = pd.read_csv(file)

                        x1 = df1['x coord'][df1["id"] == idpt][idpt]
                        y1 = df1['y coord'][df1["id"] == idpt][idpt]
                        z1 = df1['z coord'][df1["id"] == idpt][idpt]                      
                        

                        x.append(x1)
                        y.append(y1)
                        z.append(z1)

                        data = {'x': x, 'y': y, 'z': z}
                        path = pd.DataFrame(data)

                        fileName = f'output/mHRpath-{value}.csv'

                        path.to_csv(fileName, index=False)
                

                df = []
                df = []
                return True

    return False

def run_process(par2_exe, configout, value, xPlane):
    # Start the process
    process = subprocess.Popen([par2_exe, configout])

    # Monitor the folder every 10 seconds
    while True:
        if check_file("C:\\Users\\Andrea\\Documents\\PT\\output", value, xPlane):
            # Maximum value is greater than 100, terminate the process
            process.terminate()
            print("Process terminated successfully.")
            break

        time.sleep(0.3)

# p1 = (1, 1)
# p2 = (4, 1)

# grid_extents = [0, Lx, 0, Ly]
# resolution = (del_R, del_C)

# find_intersecting_points(p1, p2, grid_extents, resolution)

