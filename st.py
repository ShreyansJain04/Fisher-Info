
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# # Define the function
# def function(x):
#     return x**2 - 10 * torch.cos(2 * np.pi * x) + 10

# # Convert the function to a PyTorch tensor
# x = torch.tensor([2.0], requires_grad=True)  # Starting point

# # Define the SGD optimizer
# optimizer = torch.optim.SGD([x], lr=0.01)

# # Create a figure and axis for the animation
# fig, ax = plt.subplots()
# ax.set_xlim(-20, 20)
# ax.set_ylim(-2, 50)
# line, = ax.plot([], [], 'ro', label='Optimization Path')
# func_line, = ax.plot([], [], label='Function: f(x) = sin(x) + 0.5x')

# # Initialization function for the animation
# def init():
#     line.set_data([], [])
#     func_line.set_data([], [])
#     return line, func_line

# # Animation update function
# x_values = [2.0]
# def animate(i):
#     optimizer.zero_grad()  # Clear gradients from previous iteration
    
#     loss = function(x)  # Compute the loss
#     loss.backward()  # Compute gradients with respect to x
#     optimizer.step()  # Update x using the computed gradients
#     x_values.append(x.item())  # Store the current x value
    
#     line.set_data(x_values, [function(torch.tensor(val)).item() for val in x_values])
#     func_line.set_data(np.linspace(-6, 6, 400), function(torch.tensor(np.linspace(-6, 6, 400))).detach().numpy())
#     return line, func_line

# # Create the animation
# num_iterations = 1000
# anim = FuncAnimation(fig, animate, init_func=init, frames=num_iterations, interval=200, blit=True)

# # Show the animation
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.title('Optimization using SGD in PyTorch (Animation)')
# plt.legend()
# plt.grid(True)
# plt.show()


import streamlit as st
import numpy as np
from scipy.stats import norm, poisson, binom

# Define a dictionary of distributions and their respective parameter options
distributions = {
    'Normal': {
        'Mean': st.sidebar.slider('Mean', -10.0, 10.0, 0.0, 0.1),
        'Standard Deviation': st.sidebar.slider('Standard Deviation', 0.1, 10.0, 1.0, 0.1)
    },
    'Poisson': {
        'Lambda': st.sidebar.slider('Lambda', 0.1, 10.0, 1.0, 0.1)
    },
    'Binomial': {
        'n': st.sidebar.slider('n (Number of Trials)', 1, 100, 10, 1),
        'p': st.sidebar.slider('p (Probability of Success)', 0.0, 1.0, 0.5, 0.01)
    }
}

# Function to calculate Fisher information for a given distribution and its parameters
def calculate_fisher_info(distribution, params):
    if distribution == 'Normal':
        mean = params['Mean']
        std_dev = params['Standard Deviation']
        return 1 / (std_dev ** 2)
    elif distribution == 'Poisson':
        lambd = params['Lambda']
        return 1 / lambd
    elif distribution == 'Binomial':
        n = params['n']
        p = params['p']
        return n / (p * (1 - p))

# Main app
def main():
    st.title('Distribution and Sensitivity Analysis')
    
    # Select distribution
    selected_distribution = st.sidebar.selectbox('Select Distribution', list(distributions.keys()))
    
    # Display distribution parameters and allow user to adjust them
    params = distributions[selected_distribution]
    st.sidebar.subheader('Distribution Parameters')
    for param_name, param_value in params.items():
        st.sidebar.write(f"{param_name}: {param_value}")
    
    # Calculate and display Fisher information
    fisher_info = calculate_fisher_info(selected_distribution, params)
    st.subheader('Fisher Information')
    st.write(f"Fisher information for {selected_distribution} distribution with parameters:")
    for param_name, param_value in params.items():
        st.write(f"{param_name}: {param_value}")
    st.write(f"Fisher information: {fisher_info:.4f}")
    
    # Sensitivity analysis
    st.subheader('Sensitivity Analysis')
    st.write('Explore how the Fisher information changes with different parameter values:')
    
    # Allow user to adjust parameter values for sensitivity analysis
    for param_name, param_value in params.items():
        new_value = st.slider(f"New value for {param_name}", float(param_value) - 5.0, float(param_value) + 5.0, float(param_value), 0.1)
        params[param_name] = new_value
    
    # Calculate and display updated Fisher information
    updated_fisher_info = calculate_fisher_info(selected_distribution, params)
    st.write(f"Updated Fisher information: {updated_fisher_info:.4f}")

if __name__ == '__main__':
    main()
