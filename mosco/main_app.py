"""Renders the Mosco A/B Test Dashboard web app. Made with Streamlit. 
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
# Import the hypothesis_testing.py module
from hypothesis_testing import *


def home(logo_path, homepage_path):
    '''The home page. '''
    st.image(logo_path, use_column_width=True)
    with open(homepage_path, 'r', encoding='utf-8') as homepage:
        st.markdown(homepage.read())


def bernoulli_ttest_ui(tech_note_path):
    '''The Two-sample Student's t-test - Bernoulli variables section. '''
    with st.beta_container():
        st.title('Two-sample Student\'s t-test')
        st.header('Bernoulli variables')

    with st.beta_container():

        col1, col2 = st.beta_columns([2, 3])

        with col1:  
            visitors_1 = st.number_input(
                'Visitors A: ', min_value=0, value=80000)
            conversions_1 = st.number_input(
                'Conversions A: ', min_value=0, value=1600)
            visitors_2 = st.number_input(
                'Visitors B: ', min_value=0, value=80000)
            conversions_2 = st.number_input(
                'Conversions B: ', min_value=0, value=1696)
            conf_level = st.radio('Confidence level: ',
                                  ('0.90', '0.95', '0.99'))
            hypo_sides = st.radio('Hypothesis type: ',
                                  ('One-sided', 'Two-sided'))

        # Calculate statistics
        mu_1, mu_2, sigma_1, sigma_2 = bernoulli_stats(
            visitors_1, visitors_2, conversions_1, conversions_2)
        tstat, p_value = scipy_ttest_ind_from_stats(
            mu_1, mu_2, sigma_1, sigma_2, visitors_1, visitors_2)

        # Plot normal distribution graph
        x_1 = np.linspace(mu_1 - 3 * sigma_1, mu_1 + 3 * sigma_1, 100)
        y_1 = stats.norm.pdf(x_1, loc=mu_1, scale=sigma_1)
        x_2 = np.linspace(mu_2 - 3 * sigma_2, mu_2 + 3 * sigma_2, 100)
        y_2 = stats.norm.pdf(x_2, loc=mu_2, scale=sigma_2)

        if mu_1 < mu_2:
            annot_pos_1, annot_pos_2 = 'bottom left', 'bottom right'
        else:
            annot_pos_2, annot_pos_1 = 'bottom left', 'bottom right'

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=x_1, y=y_1, fill='tozeroy', name='Group A')
        )
        fig.add_vline(x=mu_1, line_dash='dash', annotation_text='Group A: {:.4f}'.format(mu_1),
                      annotation_position=annot_pos_1, line_width=1)
        fig.add_trace(
            go.Scatter(x=x_2, y=y_2, fill='tozeroy', name='Group B')
        )
        fig.add_vline(x=mu_2, line_dash='dash', annotation_text='Group B: {:.4f}'.format(mu_2),
                      annotation_position=annot_pos_2, line_width=1)

        with col2:
            st.write(fig)

    # Render results in tables
    with st.beta_container():

        st.text('')  # Line break
        col1, col2 = st.beta_columns([2, 1])

        with col1:
            st.markdown('''
                |                 	| Group A 	| Group B 	|Difference |
                |-----------------	|---------	|---------	|---------	|
                | Conversion rate 	| {:.4f}  	| {:.4f}  	| {:.4f}  	|
                | Standard error  	| {:.4f}  	| {:.4f}  	| {:.4f}  	|
            '''.format(mu_1, sigma_1, mu_1 - mu_2, mu_2, sigma_2, np.sqrt(sigma_1 ** 1 + sigma_2 ** 2)))

        with col2:
            st.markdown('''
                | Statistics        | Value     |
                |-----------------	|---------	|
                | t-statistic   	| {:.4f}  	|
                | p-value         	| {:.4f}  	|
            '''.format(tstat, p_value))

        if hypo_sides == 'Two-sided':
            p_threshold = (1 - float(conf_level)) / 2
        else:
            p_threshold = 1 - float(conf_level)

        st.text('')  # Line break

        if p_value < p_threshold:
            st.success('''
                Significant! 
                You can be {:.0%} confident that the difference is not a result of random chance.
            '''.format(float(conf_level)))
        else:
            st.info('''
                Not Significant. 
                There is no significant difference across two groups or you need to collect more observations.
            ''')

    # Render technical notes
    with open(tech_note_path, 'r') as tech_note:
        with st.beta_expander(label='Technical notes'):
            st.markdown(tech_note.read())


def continuous_ttest_ui():
    '''The Two-sample Student's t-test - Continuous variables section. '''
    with st.beta_container():
        st.title('Two-sample Student\'s t-test')
        st.header('Continuous variables')
    st.markdown('''
        ### 💡 思路建议

        支持两种数据输入：
        - 直接输入两组样本的均值、标准差，给出检验结果
        - 上传观测样本文件，自动计算均值、标准差，并绘制样本分布直方图/核密度估计图，给出检验结果
        ''')


def main():
    '''Add control flows to organize the UI sections. '''
    side_menu_selectbox = st.sidebar.selectbox(
        'Type',
        ('Home', '2-Sample Student\'s t-test')
    )
    if side_menu_selectbox == 'Home':
        home(logo_path='../docs/logo.png', homepage_path='../docs/homepage.md')
    if side_menu_selectbox == '2-Sample Student\'s t-test':
        sub_menu_selectbox = st.sidebar.selectbox(
            'Subtype', ('Bernoulli variables', 'Continuous variables'))
        if sub_menu_selectbox == 'Bernoulli variables':
            bernoulli_ttest_ui(tech_note_path='../docs/two_sample_ttest.md')
        elif sub_menu_selectbox == 'Continuous variables':
            continuous_ttest_ui()


if __name__ == '__main__':
    main()
