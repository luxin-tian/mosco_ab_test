"""Renders the Mosco A/B Test Dashboard web app. Made with Streamlit. 
"""
import os 
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats
# import plotly.express as px
import plotly.graph_objects as go
# import plotly.figure_factory as ff
import plotly.tools as tls
from plotly.subplots import make_subplots
import seaborn as sns
from matplotlib import pyplot as plt
# Import the hypothesis_testing.py module
from hypothesis_testing import *
import streamlit_analytics

def home(homepage_path, privacy_path, contact_path):
    '''The home page. '''
    with open(homepage_path, 'r', encoding='utf-8') as homepage: 
        homepage = homepage.read().split('---Insert video---')
        st.markdown(homepage[0], unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        with col2: 
            st.video('https://www.youtube.com/watch?v=zFMgpxG-chM')
        with col1: 
            st.image('https://images.ctfassets.net/zw48pl1isxmc/4QYN7VubAAgEAGs0EuWguw/165749ef2fa01c1c004b6a167fd27835/ab-testing.png', use_column_width='auto')
            st.text('Image source: Optimizely')
        st.markdown(homepage[1], unsafe_allow_html=True)
    contact_us_ui(contact_path, if_home=True)
    with st.beta_expander(label='Privacy Notice'): 
        with open(privacy_path, 'r', encoding='utf-8') as privacy: 
            st.markdown(privacy.read())


def contact_us_ui(contact_path, if_home=False): 
    if not if_home: 
        st.write('# New Features 💡')
        st.text_input('Send us suggestions', 'Write something...')
        if_send = st.button('Send')
        if if_send: 
            st.success('Thank you:) Your suggestions have been received. ')
            st.balloons()
    with open(contact_path, 'r', encoding='utf-8') as contact: 
        st.write(contact.read())


def ttest_plot(mu_1, mu_2, sigma_1, sigma_2, conf_level, tstat, p_value, tstat_denom, hypo_type, observed_power): 
    # Plot normal distribution graph
    x_1 = np.linspace(mu_1 - 3 * sigma_1, mu_1 + 3 * sigma_1, 1000)
    y_1 = scipy.stats.norm.pdf(x_1, loc=mu_1, scale=sigma_1)
    x_2 = np.linspace(mu_2 - 3 * sigma_2, mu_2 + 3 * sigma_2, 1000)
    y_2 = scipy.stats.norm.pdf(x_2, loc=mu_2, scale=sigma_2)
    x_t = np.linspace(-3, 3, 1000)
    y_t = scipy.stats.norm.pdf(x_t, loc=0, scale=1) # Use normal distribution as an approximation of t distribution. 

    if mu_1 < mu_2:
        annot_pos_1, annot_pos_2 = 'bottom left', 'bottom right'
    else:
        annot_pos_2, annot_pos_1 = 'bottom left', 'bottom right'

    critical_val = {'0.99': 2.33, '0.95': 1.96, '0.90': 1.64}[conf_level]

    fig = make_subplots(rows=2, cols=1)
    fig.add_trace(
        go.Scatter(x=x_1, y=y_1, fill='tozeroy', name='Group A'), row=1, col=1
    )
    fig.add_vline(x=mu_1, line_dash='dash', annotation_text='Group A: {:.4f}'.format(mu_1),
                annotation_position=annot_pos_1, line_width=1, row=1, col=1)
    fig.add_trace(
        go.Scatter(x=x_2, y=y_2, fill='tozeroy', name='Group B'), row=1, col=1
    )
    fig.add_vline(x=mu_2, line_dash='dash', annotation_text='Group B: {:.4f}'.format(mu_2),
                annotation_position=annot_pos_2, line_width=1, row=1, col=1)
    fig.add_trace(
        go.Scatter(x=x_t, y=y_t, name='t-test'), row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=x_t, y=np.where(abs(x_t) < critical_val, 0, y_t), line_dash='dash', fill='tozeroy', name='Rejection region'), row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=x_t, y=np.where(abs(x_t) >= critical_val, 0, y_t), line_dash='dash', fill='tozeroy', name='Non-rejection region'), row=2, col=1
    )
    fig.add_vline(x=tstat, line_dash='dash', annotation_text='t-statistic: {:.4f}<br>p-value: {:.4f}'.format(tstat, p_value), annotation_align='left', line_width=1, row=2, col=1)

    # Render test results
    with st.beta_container(): 

        if hypo_type == 'Two-sided':
            p_threshold = (1 - float(conf_level)) / 2
        else:
            p_threshold = 1 - float(conf_level)

        # Render message box
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

        # Render plots
        st.write(fig)

        # Render results in tables
        st.markdown('''
            |                 	| Group A 	| Group B 	|Difference |
            |-----------------	|---------	|---------	|---------	|
            | Conversion rate 	| {:.4f}  	| {:.4f}  	| {:.4f}  	|
            | Standard error  	| {:.4f}  	| {:.4f}  	| {:.4f}  	|
            | t-statistic     	|         	|        	| {:.4f}  	|
            | p-value         	|         	|        	| {:.4f}  	|
            | Observed power  	|         	|        	| {:.4f}  	|
        '''.format(mu_1, mu_2, mu_1 - mu_2, sigma_1, sigma_2, tstat_denom, tstat, p_value, observed_power))


def bernoulli_ttest_ui(tech_note_path):
    '''The Two-sample Student's t-test - Bernoulli variables section. '''

    # Render the header. 
    with st.beta_container():
        st.title('Two-sample Student\'s t-test')
        st.header('Bernoulli variables')
        st.info('If the outcome variable is binary, for example, if you are testing for Click-through Rates (CTR) or customer retention rates, we suggest using a Chi-squared test. ')

    # Render input widgets
    with st.beta_container():
        col1, col2 = st.columns([1, 1])
        with col1:  
            st.subheader('Group A')
            visitors_1 = st.number_input(
                'Visitors A: ', min_value=10, value=80000)
            conversions_1 = st.number_input(
                'Conversions A: ', min_value=0, value=1600)
            conf_level = st.select_slider('Confidence level: ', ('0.90', '0.95', '0.99'))
        with col2: 
            st.subheader('Group B')
            visitors_2 = st.number_input(
                'Visitors B: ', min_value=10, value=80000)
            conversions_2 = st.number_input(
                'Conversions B: ', min_value=0, value=1696)
            hypo_type = st.radio('Hypothesis type: ', ('One-sided', 'Two-sided'))
    
        # Assert visitors>=conversions
        try: 
            assert visitors_1 >= conversions_1 
        except: 
            raise ValueError('Visitors must be more the the converted. ')

        # Calculate statistics
        mu_1, mu_2, sigma_1, sigma_2 = bernoulli_stats(
            visitors_1, visitors_2, conversions_1, conversions_2)
        tstat, p_value, tstat_denom, pooled_sd, effect_size = scipy_ttest_ind_from_stats(
            mu_1, mu_2, sigma_1, sigma_2, visitors_1, visitors_2)
        observed_power = sm_tt_ind_solve_power(effect_size=effect_size, n1=visitors_1, n2=visitors_2, alpha=1-float(conf_level), power=None, hypo_type=hypo_type, if_plot=False)

        # Render the results
        ttest_plot(mu_1, mu_2, sigma_1, sigma_2, conf_level, tstat, p_value, tstat_denom, hypo_type, observed_power)

    # Render technical notes
    with st.beta_expander(label='Technical notes'):
        with open(tech_note_path, 'r') as tech_note:
            st.markdown(tech_note.read())


def ttest_power_ui():
    '''The Power Analysis for Two-sample Student's t-test section. '''

    # Render the header. 
    with st.beta_container():
        st.title('Two-sample Student\'s t-test')
        st.header('Power analysis')

    # Render input boxes and plots
    with st.beta_container():
        param_list = ['Standardized effect size', 'alpha (1 - Significance level)', 'Power (1 - beta)', 'Group A sample size']
        param_name = {'Standardized effect size': 'effect_size', 'alpha (1 - Significance level)': 'alpha', 'Power (1 - beta)': 'power', 'Group A sample size': 'n1'}
        param_default = { 
            'Standardized effect size': {'min_value': 0.0, 'value': 0.2}, 
            'alpha (1 - Significance level)': {'min_value': 0.01, 'max_value': 0.5, 'value': 0.05}, 
            'Power (1 - beta)': {'min_value': 0.01, 'max_value': 0.99, 'value': 0.8}, 
            'Group A sample size': {'min_value': 10, 'value': 1000}}
        kwargs = {}
        unknown_param = st.selectbox('Solve for: ', param_list)
        param_list.remove(unknown_param)
        for param in param_list: 
            kwargs[param] = st.number_input(param, **param_default[param])
        kwargs = {param_name[param]: kwargs[param] for param in param_list}
        kwargs['ratio'] = st.number_input('Sample size Ratio (Group B : Group A)', min_value=0.01, value=1.0)
        try: 
            value, fig = sm_tt_ind_solve_power(**kwargs)
        except: 
            st.exception('The solution is out of bound. Please adjust the input parameters. ')
            st.stop()
        st.success('{}: `{:.4f}`'.format(unknown_param, value))
        st.write(fig, width=200)


def continuous_ttest_from_stats_ui():
    '''The Two-sample Student's t-test - Continuous variables (from statistics) section. '''
    
    # Render the header. 
    with st.beta_container():
        st.title('Two-sample Student\'s t-test')
        st.header('Continuous variables')

    # Render input boxes and plots
    with st.beta_container():
        col1, col2 = st.columns([1, 1])
        with col1:  
            st.subheader('Group A')
            visitors_1 = st.number_input('Sample size A: ', value=80000)
            mu_1 = st.number_input('Sample mean A: ', value=64.5)
            sigma_1 = st.number_input('Sample standard deviation A: ', min_value=0.0, value=30.8)
            conf_level = st.select_slider('Confidence level: ', ('0.90', '0.95', '0.99'))
        with col2: 
            st.subheader('Group B')
            visitors_2 = st.number_input('Sample size B: ', value=80000)
            mu_2 = st.number_input('Sample mean B: ', value=68.7)
            sigma_2 = st.number_input('Sample standard deviation B: ', min_value=0.0, value=38.1)
            hypo_type = st.radio('Hypothesis type: ', ('One-sided', 'Two-sided'))     

    # Calculate statistics
    tstat, p_value, tstat_denom, pooled_sd, effect_size = scipy_ttest_ind_from_stats(
        mu_1, mu_2, sigma_1, sigma_2, visitors_1, visitors_2)
    observed_power = sm_tt_ind_solve_power(effect_size=effect_size, n1=visitors_1, n2=visitors_2, alpha=1-float(conf_level), power=None, hypo_type=hypo_type, if_plot=False)

    # Render the results
    ttest_plot(mu_1, mu_2, sigma_1, sigma_2, conf_level, tstat, p_value, tstat_denom, hypo_type, observed_power)

    # # Render technical notes
    # with st.beta_expander(label='Technical notes'):
    #     with open(tech_note_path, 'r') as tech_note:
    #         st.markdown(tech_note.read())


@st.cache(show_spinner=False)
def _load_data(filepath): 
    return pd.read_csv(filepath)


def _process_data(df, col, if_dropna, if_remove_outliers, outlier_lower_qtl, outlier_upper_qtl): 
    if if_dropna: 
        df = df.dropna(subset=[col])
    if if_remove_outliers: 
        q_low = df[col].quantile(outlier_lower_qtl)
        q_hi  = df[col].quantile(outlier_upper_qtl)
        df = df[(df[col] < q_hi) & (df[col] > q_low)]
    return df

def ttest_upload_data_ui(): 
    '''The Two-sample Student's t-test - Continuous variables (upload data) section. '''
    
    # Render the header. 
    with st.beta_container(): 
        st.title('Two-sample Student\'s t-test')
        st.header('Continuous variables')
    
    # Render file dropbox
    with st.beta_expander('Upload data', expanded=True): 
        how_to_load = st.selectbox('How to access raw data? ', ('Upload', 'URL', 'Sample data'))
        if how_to_load == 'Upload': 
            uploaded_file = st.file_uploader("Choose a CSV file", type='.csv')
        elif how_to_load == 'URL': 
            uploaded_file = st.text_input('File URL: ')
            if uploaded_file == '': 
                uploaded_file = None
        elif how_to_load == 'Sample data': 
            uploaded_file = 'https://raw.githubusercontent.com/luxin-tian/mosco_ab_test/main/sample_data/cookie_cats.csv'
        if uploaded_file is not None: 
            with st.spinner('Loading data...'): 
                df = _load_data(uploaded_file)
    
    if uploaded_file is not None: 
        with st.beta_expander('Data preview', expanded=True): 
            with st.spinner('Loading data...'): 
                st.dataframe(df)
                st.write('`{}` rows, `{}` columns'.format(df.shape[0],df.shape[1]))
    
    if uploaded_file is not None: 
        with st.beta_expander('Configurations', expanded=True): 
            df_columns_types = [ind + ' (' + val.name + ')' for ind, val in df.dtypes.iteritems()]
            df_columns_dict = {(ind + ' (' + val.name + ')'): ind for ind, val in df.dtypes.iteritems()}
            var_group_label = df_columns_dict[st.selectbox('Group label', df_columns_types)]
            col1, col2 = st.columns(2) 
            with col1:
                var_group_name_1 = st.selectbox('Group name A', df[var_group_label].unique())
            with col2:
                var_group_name_2 = st.selectbox('Group name B', df[var_group_label].unique())
            var_outcome = [df_columns_dict[var] for var in st.multiselect('Outcome variable: ', df_columns_types)]
            col1, col2 = st.columns([1, 1])
            with col1: 
                conf_level = st.select_slider('Confidence level: ', ('0.90', '0.95', '0.99'))
            with col2: 
                hypo_type = st.radio('Hypothesis type: ', ('One-sided', 'Two-sided'))
            if_dropna = st.checkbox('Drop null values', value=True)
            if_remove_outliers = st.checkbox('Remove outliers', value=False)
            if if_remove_outliers: 
                outlier_lower_qtl, outlier_upper_qtl = st.slider('Quantiles (observations falling into the tails will be removed): ', min_value=0.0, max_value=1.0, step=0.01, value=(0.0, 0.95))
                # col1, col2 = st.columns(2) 
                # with col1: 
                #     outlier_lower_qtl = st.slider('Lower quantile: ', min_value=0.0, max_value=0.25, step=0.01, value=0.0)
                # with col2: 
                #     outlier_upper_qtl = st.slider('Upper quantile: ', min_value=0.75, max_value=1.00, step=0.01, value=0.99)
            else: 
                outlier_lower_qtl, outlier_upper_qtl = None, None
            if_data_description = st.checkbox('Show descriptive statistics', value=False)
            if_apply = st.button('Confirm')
    
    if uploaded_file is not None: 
        if if_apply: 
            if var_group_name_1 == var_group_name_2: 
                st.error('The names of Group A and Group B cannot be identical. ')
                st.stop()
            for col in var_outcome: 
                df = _process_data(df=df, col=col, if_dropna=if_dropna, if_remove_outliers=if_remove_outliers, outlier_lower_qtl=outlier_lower_qtl, outlier_upper_qtl=outlier_upper_qtl)
            # Render hypothesis testing
            with st.beta_expander('Hypothesis testing', expanded=True): 
                with st.spinner('Calculating...'): 
                    df_group_1 = df[df[var_group_label] == var_group_name_1]
                    df_group_2 = df[df[var_group_label] == var_group_name_2]
                    for var in var_outcome: 
                        st.markdown(f'`{var}`: {df[var].dtype}')
                        mu_1 = np.mean(df_group_1[var])
                        mu_2 = np.mean(df_group_2[var])
                        sigma_1 = np.std(df_group_1[var], ddof=1)
                        sigma_2 = np.std(df_group_2[var], ddof=1)
                        n_1 = len(df_group_1[var])
                        n_2 = len(df_group_2[var])

                        tstat, p_value, tstat_denom, pooled_sd, effect_size = scipy_ttest_ind_from_stats(
                            mu_1, mu_2, sigma_1, sigma_2, n_1, n_2)
                        observed_power = sm_tt_ind_solve_power(effect_size=effect_size, n1=n_1, n2=n_2, alpha=1-float(conf_level), power=None, hypo_type=hypo_type, if_plot=False)

                        # Render the results
                        ttest_plot(mu_1, mu_2, sigma_1, sigma_2, conf_level, tstat, p_value, tstat_denom, hypo_type, observed_power)

            # Render descriptive statistics
            if if_data_description: 
                with st.beta_expander('Data descriptions', expanded=True): 
                    with st.spinner('Processing data...'): 
                        # if if_factorize:  
                        #     df[var_hot_encoding] = df[var_hot_encoding].astype('category')
                        df = df[(df[var_group_label] == var_group_name_1) | (df[var_group_label] == var_group_name_2)]
                        df_summary = df.groupby(by=var_group_label).describe(include='all')

                        # Plot distribution
                        for var in var_outcome: 
                            st.markdown(f'`{var}`: {df[var].dtype}')
                            st.table(df_summary[var].T.dropna())
                            fig_1 = sns.displot(data=df, x=var, col=var_group_label, kde=True)
                            fig_2 = sns.displot(data=df, kind="ecdf", x=var, hue=var_group_label, rug=True)
                            fig_3, ax = plt.subplots()
                            ax = sns.boxplot(data=df, y=var, hue=var_group_label)
                            st.pyplot(fig_1)
                            col1, col2 = st.columns([1, 1.1])
                            with col1: 
                                st.pyplot(fig_2)
                            with col2: 
                                st.pyplot(fig_3)
    return 


def chi_squared_ui(): 
    '''The Two-sample Chi-squred test. '''
    
    # Render the header. 
    with st.beta_container(): 
        st.title('Chi-Squared Test')
        st.info('This section is under development and is pre-released for test. ')

    # Render input widgets
    with st.beta_container():
        col1, col2 = st.columns([1, 1])
        with col1:  
            st.subheader('Group A')
            visitors_1 = st.number_input(
                'Visitors A: ', min_value=10, value=80000)
            conversions_1 = st.number_input(
                'Conversions A: ', min_value=0, value=1600)
            conf_level = st.select_slider('Confidence level: ', ('0.90', '0.95', '0.99'))
        with col2: 
            st.subheader('Group B')
            visitors_2 = st.number_input(
                'Visitors B: ', min_value=10, value=80000)
            conversions_2 = st.number_input(
                'Conversions B: ', min_value=0, value=1696)
            hypo_type = st.radio('Hypothesis type: ', ('One-sided', 'Two-sided'))
    
    # Assert visitors>=conversions
    if_apply = st.button('Confirm')
    if if_apply or ((visitors_1 == 80000) * (visitors_2 == 80000) * (conversions_1 == 1600) * (conversions_2 == 1696)): 
        try: 
            assert visitors_1 >= conversions_1 
        except: 
            raise ValueError('Visitors must be more the the converted. ')

        # Calculate statistics
        chi2, p_value, expected, observed, fig = scipy_chi2_from_stats(visitors_1, visitors_2, conversions_1, conversions_2)

        # Render the results 
        with st.beta_container(): 
            if hypo_type == 'Two-sided':
                p_threshold = (1 - float(conf_level)) / 2
            else:
                p_threshold = 1 - float(conf_level)

            # Render message box
            if p_value < p_threshold:
                st.success('''
                    Significant! 
                    You can be {:.0%} confident that the difference is not a result of random chance - the observed frequencies seem not to be independent.
                '''.format(float(conf_level)))
            else:
                st.info('''
                    Not Significant. 
                    There is no significant difference across two groups - the observed frequencies seem to be independent.
                ''')

            col1, col2 = st.columns([2, 1])
            
            with col1: 
                st.pyplot(fig)

            st.markdown('''
                |                 	    | Group A 	| Group B 	|Difference |
                |-----------------	    |---------	|---------	|---------	|
                | Non-converted    	    | {:.0f}  	| {:.0f}  	| {:.0f}  	|
                | Converted             | {:.0f}  	| {:.0f} 	| {:.0f}  	|
                | chi-squred            |         	|        	| {:.4f}  	|
                | p-value         	    |         	|        	| {:.4f}  	|
            '''.format(visitors_1 - conversions_1, visitors_2 - conversions_2, 
            visitors_1 - conversions_1 - visitors_2 + conversions_2, 
            conversions_1, conversions_2, conversions_1 - conversions_2, 
            chi2, p_value)
            )

    # Render technical notes
    with st.beta_expander(label='References'):
        st.markdown('- [A/B Testing with Chi-Squared Test to Maximize Conversions and CTRs](https://towardsdatascience.com/a-b-testing-with-chi-squared-test-to-maximize-conversions-and-ctrs-6599271a2c31)')


def main():
    '''Add control flows to organize the UI sections. '''
    st.sidebar.image('./docs/logo.png', width=250)
    st.sidebar.write('') # Line break
    st.sidebar.header('Navigation')
    side_menu_selectbox = st.sidebar.radio(
        'Menu', ('Home', '2-Sample Student\'s T-Test', 'Chi-Squared Test', 'Suggest New Features'))
    if side_menu_selectbox == 'Home':
        home(homepage_path='./docs/homepage.md', privacy_path='./docs/privacy_notice.md', contact_path='./docs/contact_us.md')
    elif side_menu_selectbox == '2-Sample Student\'s T-Test':
        sub_menu_selectbox = st.sidebar.radio(
            'Data source', ('With raw data', 'With statistics', 'Power analysis'))
        if sub_menu_selectbox == 'With raw data': 
            ttest_upload_data_ui()
        elif sub_menu_selectbox == 'With statistics': 
            sub_sub_menu_select_box = st.sidebar.radio(
                'Options', ('Continuous variable', 'Binary variable'))
            if sub_sub_menu_select_box == 'Continuous variable': 
                continuous_ttest_from_stats_ui()
            elif sub_sub_menu_select_box == 'Binary variable': 
                bernoulli_ttest_ui(tech_note_path='./docs/two_sample_ttest.md')
        elif sub_menu_selectbox == 'Power analysis': 
            ttest_power_ui()
    elif side_menu_selectbox == 'Suggest New Features':
        contact_us_ui(contact_path='./docs/contact_us.md')
    elif side_menu_selectbox == 'Chi-Squared Test': 
        chi_squared_ui()


# @st.cache(allow_output_mutation=True, persist=True)
# def visitor_analytics():
#     return {}


# def pwd_auth(): 
#     st.info('MOSCO is currently under development. Please enter your tokens if you are a developer. ')
#     st.header('Authentication')
#     visitor_cnt = visitor_analytics()
#     username = st.text_input('Username', value='')
#     if username == '': 
#         st.info('Please input your username. ')
#         st.stop()
#     st.markdown(f'You will login as `{username}`. ')
#     pwd = st.text_input('Trial token', type='password')
#     if pwd == 'MOSCO-2020' or pwd == 'mosco2020admin': 
#         st.success('Success. Welcome back :)')
#         visitor_cnt[username] = visitor_cnt.get(username, [])
#         visitor_cnt[username].append(str(datetime.now()))
#         if username in ['yezi-li', 'luxin-tian'] and pwd == 'mosco2020admin': 
#             st.text('Visitor analytics')
#             st.json(visitor_cnt)
#         main()
#     else: 
#         if pwd != '': 
#             st.error('Invalid trial token. Please contact the development team to request for a trial token. ')
#         else: 
#             st.stop()


if __name__ == '__main__': 
    import os
    with streamlit_analytics.track(unsafe_password=os.environ['analytics_pwd']): 
        st.set_page_config(page_title='MOSCO - A/B Test Toolkits', page_icon='./docs/icon.png', layout='centered', initial_sidebar_state='auto')
        try: 
            main()
        except: 
            st.error('Oops! Something went wrong...Please check your input.\nIf you think there is a bug, please open up an [issue](https://github.com/luxin-tian/mosco_ab_test/issues) and help us improve. ')
            raise
