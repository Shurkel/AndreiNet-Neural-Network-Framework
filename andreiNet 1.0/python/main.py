import matplotlib.pyplot as plt

# Read points from file
def read_points_from_file(filename):
    points = []
    with open(filename, 'r') as file:
        for line in file:
            study_hours, sleep_hours, pass_fail, color = line.strip().split(',')
            points.append((float(study_hours), float(sleep_hours), int(pass_fail), color))
    return points

# Plot points
def plot_points(points):
    fig, ax = plt.subplots()
    for study_hours, sleep_hours, pass_fail, color in points:
        if pass_fail == 1:
            marker = 'o'
            label = 'Pass'
        else:
            marker = 'x'
            label = 'Fail'
        ax.scatter(study_hours, sleep_hours, c=color, marker=marker, label=label)
    
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    ax.set_xlabel('Study Hours')
    ax.set_ylabel('Sleep Hours')
    ax.set_title('Student Performance')
    plt.show()

# Main function
if __name__ == '__main__':
    filename = '../python/data.txt'  # Specify the filename
    points = read_points_from_file(filename)
    plot_points(points)
