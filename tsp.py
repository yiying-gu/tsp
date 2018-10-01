
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import os


def read_numbers(data_file):
    input_data_file = open(data_file, 'r')
    input_data = input_data_file.readlines()
    input_data_file.close()

    numbers = np.array([])
    for i_line in xrange(len(input_data)):
        entries = input_data[i_line].split()
        entries = filter(None, entries) # remove empty entries
        line_numbers = [ float(x) if x.lower != "inf" else float("inf") for x in entries ]
        numbers = np.append(numbers, line_numbers)
    return numbers


def read_data(data_file):
    numbers = read_numbers(data_file)
    cur_entry = 0

    # number of points
    num_points = int(numbers[cur_entry])
    cur_entry += 1

    # get data on the points
    points = np.zeros((num_points, 2))
    for i_point in xrange(num_points):
        points[i_point, 0] = float(numbers[cur_entry])
        cur_entry += 1
        points[i_point, 1] = float(numbers[cur_entry])
        cur_entry += 1

    return points


def dist(A, B):
    return math.sqrt( (A[0] - B[0]) * (A[0] - B[0]) + (A[1] - B[1]) * (A[1] - B[1]) )


def check_tsp_solution( solution, points ):
    num_points = points.shape[0]
    visited_nodes = np.zeros(num_points, dtype=bool)
    path_length = dist( points[solution[0]], points[solution[-1]] )
    for i_point in xrange(num_points-1):
        visited_nodes[i_point] = True
        path_length += dist( points[solution[i_point]], points[solution[i_point+1]] )

    is_valid_solution = False in visited_nodes
    return is_valid_solution, path_length


def plot_tsp_solution(solution, points):
    is_valid_solution, path_length = check_tsp_solution( solution, points )

    x = np.hstack((points[solution][:,0], points[solution[0]][0]))
    y = np.hstack((points[solution][:,1], points[solution[0]][1]))
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    solution_quality = ['Inconsistent', 'Valid']
    plt.title( '%s solution; %d points; length = %f'%(solution_quality[is_valid_solution], len(points), path_length) )
    plt.show(block=True)

"""
def make_dummy_solution(points):
    num_points = points.shape[0]
    solution = np.arange(num_points)
    solution_value = dist( points[0], points[-1] )
    for i_point in xrange(num_points-1):
        solution_value += dist( points[i_point], points[i_point+1] )

    for i in xrange (num_points + 1):
        dfs(points)
        solution_value = min_length
        
    return solution_value, solution
"""

two_opt_swap = lambda r,i,k: np.concatenate((r[0:i],r[k:-len(r)+i-1:-1],r[k+1:len(r)]))
    
    
def two_opt(points, improvement_threshold):
    num_points = points.shape[0]
    solution = np.arange(num_points)
    solution_value = dist(points[0],points[-1])
    for i_point in xrange(num_points-1):
        solution_value += dist( points[i_point], points[i_point+1] )

    for i in xrange(num_points-1):
        min_distance = 1000000
        min_pos = 0
        for j in xrange(i+1, num_points):
            local_distance = dist(points[i],points[j])
            if local_distance < min_distance:
                min_distance = local_distance
                min_pos = j
            new_route1 = two_opt_swap(solution,i,min_pos)
            new_distance1 = dist(points[0],points[-1])
            for i_point in xrange(len(new_route1)-1):
                new_distance1 += dist( points[new_route1[i_point]], points[new_route1[i_point+1]] )
    solution = new_route1
    solution_value = new_distance1
          
    improvement_factor = 1
    best_distance = solution_value
    while improvement_factor > improvement_threshold:
        distance_to_beat = best_distance
        for swap_first in range(1, num_points-3):
            for swap_last in range(swap_first+1, num_points-1):
                new_route2 = two_opt_swap(solution,swap_first,swap_last)
                new_distance2 = dist(points[0],points[-1])
                for i_point in xrange(len(new_route2)-1):
                    new_distance2 += dist( points[new_route2[i_point]], points[new_route2[i_point+1]] )
                
                if new_distance2 < best_distance:
                    solution = new_route2
                    best_distance = new_distance2
                    
        improvement_factor = 1 - best_distance/distance_to_beat

    return best_distance, solution
    

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        points = read_data(file_location)
        out_file_path = os.path.splitext(file_location)[0] + "_solution.txt"

        solution_value, solution = two_opt(points, 0.01)

        print solution_value
        print ' '.join(map(str, solution))

        plot_tsp_solution(solution, points)

    else:
        print 'This script requires an input file as command line argument.'
