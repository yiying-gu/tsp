
import numpy as np
import math
import matplotlib.pyplot as plt
import sys


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
    num_points = points.shape[0]        #shape reads the dimension of the array, in this case, the first dimension of points
    visited_nodes = np.zeros(num_points, dtype=bool)
    path_length = dist( points[solution[0]], points[solution[-1]] ) #last point of solution
    for i_point in xrange(num_points-1):    #xrange varies from 0 to n-1, in this case, from 0 to num_point-1-1
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

def find_path(j):
    path_vertexs.append(j)  
    row=distance[j]  
    copy_row=[value for value in row]
    walked_vertex=[]
    for i in path_vertexs:
        walked_vertex.append(copy_row[i])
    for vertex in walked_vertex:
        copy_row.remove(vertex)
        
    if len(path_vertexs)<points.shape[0]:
        min_e=min(copy_row)
        j=row.index(min_e)
        path_length.append(min_e)
        find_path(j)
    else:
        min_e=distance[j][0]
        path_length.append(min_e)
        path_vertexs.append(0)
    
    return path_length, path_vertexs


'''
def make_dummy_solution(points):
    num_points = points.shape[0]
    solution = np.arange(num_points)
    
    solution = path_vertexs
    solution_value = path_length  
    
    solution_value = dist( points[0], points[-1] )
    for i_point in xrange(num_points-1):
        solution_value += dist( points[i_point], points[i_point+1] )

    return solution
'''

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        points = read_data(file_location)
        num_points = points.shape[0]
        distance=[[0 for i in range(num_points)] for i in range(num_points)]

        for i in xrange(num_points):
            for k in xrange(num_points):
                distance[i][k]=dist( points[i], points[k] )
        path_length=[]
        path_vertexs=[]   
        solution_value, solution =find_path(0)

        plot_tsp_solution(solution, points)
        with open("task3_test1_solution.txt",'w') as f:
            f.write(str(solution_value)+'\n')
            for i in solution:
                f.write(str(i)+'\t')

        print solution_value
        print ' '.join(map(str, solution))
    else:
        print 'This script requires an input file as command line argument.'
