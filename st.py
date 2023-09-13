

import streamlit as st
import numpy as np
from scipy.stats import norm, poisson, binom

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

    selected_distribution = st.sidebar.selectbox('Select Distribution', list(distributions.keys()))
    
    params = distributions[selected_distribution]
    st.sidebar.subheader('Distribution Parameters')
    for param_name, param_value in params.items():
        st.sidebar.write(f"{param_name}: {param_value}")
    

    fisher_info = calculate_fisher_info(selected_distribution, params)
    st.subheader('Fisher Information')
    st.write(f"Fisher information for {selected_distribution} distribution with parameters:")
    for param_name, param_value in params.items():
        st.write(f"{param_name}: {param_value}")
    st.write(f"Fisher information: {fisher_info:.4f}")

    st.subheader('Sensitivity Analysis')
    st.write('Explore how the Fisher information changes with different parameter values:')

    for param_name, param_value in params.items():
        new_value = st.slider(f"New value for {param_name}", float(param_value) - 5.0, float(param_value) + 5.0, float(param_value), 0.1)
        params[param_name] = new_value
    
    updated_fisher_info = calculate_fisher_info(selected_distribution, params)
    st.write(f"Updated Fisher information: {updated_fisher_info:.4f}")

if __name__ == '__main__':
    main()
