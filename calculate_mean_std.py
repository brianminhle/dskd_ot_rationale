import statistics

# Example list of numbers
numbers = [33.9487, 33.6003, 33.2011, 33.6379, 33.3766]

# Calculate mean and standard deviation
mean = round(statistics.mean(numbers), 2)
std_dev = round(statistics.stdev(numbers), 2)  # Use statistics.pstdev() for population standard deviation

print(f"Mean: {mean}")
print(f"Standard Deviation: {std_dev}")