from audioop import reverse
from dataclasses import dataclass
from dis import show_code
from functools import total_ordering
import itertools

import copy

import getpass
import urllib.parse

import psycopg2

import datetime
import pytz
import calendar

from math import floor, ceil

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import seaborn as sns

sns.set_theme()



class transformer:

    capacity_data = None
    last_updated = None
    secondary_voltage = None
    rating_kva = None

    def __init__(self, grid, id, power_factor = 0.98, capacity_budget = 1.0):
        self.id = id
        self.power_factor = power_factor
        self.grid = grid
        self.capacity_budget = capacity_budget

    def __str__(self):
        return f'Transformer: ({self.grid}, {self.id})'
        
    def __repr__(self):
        return f'Transformer: ({self.grid}, {self.id})'

    def retrieve_data(self, conn):
        cur = conn.cursor()
        
        cur.execute("SELECT meta_data.key, meta_data.value \
                            FROM grid_element ge \
                            JOIN json_each_text(ge.meta::json) meta_data \
                                ON true \
                            WHERE ge.grid_element_id = %s \
                                AND ge.grid_id = %s\
                                AND (meta_data.key = 'secondary_voltage' \
                                    OR meta_data.key = 'rating_kva');", (self.id, self.grid))

        fet = cur.fetchall()

        self.secondary_voltage = int(fet[1][1])
        self.rating_kva = float(fet[0][1])
        
        cur.close()

        conn.rollback()

    
        self.transformer_aggregate(conn)

    

    def update_data(self):
        return None

    def view_in_range(self, start_date, end_date):
        '''Returns a copy of the capacity data for the given date range.'''


        if self.capacity_data is None:
            return None
        
        else:

            st = datetime.datetime.fromisoformat(start_date)
            et = datetime.datetime.fromisoformat(end_date)


            return self.capacity_data[(self.capacity_data.index >= st) & (self.capacity_data.index <= et)].copy(deep = True)

    def transformer_aggregate(self, conn):
        ''' Computes load and excess capacity across the entire lifespan of the data, or within a specified time range.
            If start_date is not supplied, get all data up to end_date. If end_date is not supplied, get all data from start_date
            to now. If neither are supplied, get all data.
            
                    Parameters:
                        conn (psycopg2.connection): connection to database.
                        start_date (str or datetime object): start date of range to consider.
                        end_date (str or datetime object): end date of range to consider.

                    

            Will only recompute data if the needed data has not been queried before.
        
        '''
        if self.capacity_data is not None:
            return self.capacity_data
        
        #Collect all downstream transformers and aggregate on them.
        
        cur = conn.cursor()

        cur.execute("SELECT grid_element_id AS g_id \
                        FROM grid_get_downstream(%s, %s, false) \
                        WHERE type = 'Transformer';", (self.grid,self.id))
        
        dt_sql = pd.DataFrame(cur.fetchall(), columns = ['g_id'])


        downstream_transformer_data = []
        
        if dt_sql.shape[0] != 0:
        
            downstream_transformers = dt_sql.DataFrame()['g_id'].tolist()
        
            for transformer in downstream_transformers:
                
                downstream_t = transformer(transformer, self.power_factor)

                dt = downstream_t.transformer_aggregate()['load']
                downstream_transformer_data.append(dt)
        
        else:
            downstream_transformers = []
            
            
        # Next, collect all meters on this transformer and aggregate on them.
        
        # grid_get_downstream gets ALL meters downstream, not just those that are directly connected
        #    to this transformer. So, we'll need to iterate through all downstream transformers and remove those
        #    from our meter list.
        


        cur.execute("SELECT grid_element_id \
                        FROM grid_get_downstream(%s, %s, false) \
                        WHERE type = 'Meter';", (self.grid,self.id))
        
        meters = pd.DataFrame(cur.fetchall(), columns = ['grid_element_id']).grid_element_id.tolist()
        
        meters_to_exclude = []
        
        for transformer in downstream_transformers:
            cur.execute("SELECT grid_element_id \
                                FROM grid_get_downstream('awefice', %s, false) \
                                WHERE type = 'Meter';", transformer)
            
            meters_to_exclude = meters_to_exclude + pd.DataFrame(cur.fetchall(), columns = ['grid_element_id']).grid_element_id.tolist()

        cur.close()
        conn.rollback()
            
        meters = [m for m in meters if m not in meters_to_exclude]
        
        
        aggregate_load = None
        
        if len(meters) > 0:
            aggregate_load = pd.DataFrame(self.meter_agg(meters, conn), columns = ['load'])
            
            if len(downstream_transformer_data) > 0:
                for i in range(0, len(downstream_transformer_data)):
                    downstream_transformer_data[i] = downstream_transformer_data[i].rename('load'+ str(i))
                
                aggregate_load = aggregate_load.join(downstream_transformer_data)
                aggregate_load = pd.DataFrame(aggregate_load.sum(axis = 1), columns = ['load'])
            
            
        else: 
            
            if len(downstream_transformer_data) == 0:
                ### TODO: change this to initialize a column of 0s for load. The problem is getting the list of timestamps.
                
                raise Exception('No data to aggregate: no downstream meters or transformers.')
                
            else:
                for i in range(0, len(downstream_transformer_data)):
                    downstream_transformer_data[i] = downstream_transformer_data[i].rename('load'+ str(i))
                
                aggregate_load = pd.DataFrame(downstream_transformer_data[0])
                
                aggregate_load = aggregate_load.join(downstream_transformer_data[1:])
                aggregate_load = pd.DataFrame(aggregate_load.sum(axis = 1), columns = ['load'])
                
        
        aggregate_load['Excess Capacity'] = int(self.rating_kva)*(self.power_factor) - aggregate_load['load']
        aggregate_load['Budgeted Capacity'] = aggregate_load['Excess Capacity']*self.capacity_budget
        
        self.capacity_data = aggregate_load
        self.last_updated = datetime.datetime.now()

        return aggregate_load


    def meter_agg(self, meters, conn):
        '''Takes a list of grid elements and aggregates their load.
    
                 Parameters:
                         meters (List of str): list of grid_element_ids of all meters to aggregate
                         transformer_data (List of DataFrames): list of pandas databases containing the aggregated 
                                                                     data for connected transformers
                                                                     
                                                                     Dataframes will contain 3 columns:
                                                                         timestamp
                                                                         load in kWh
                                                                         
                         start_date (str): start date of time series range
                         end_date (str): end date of time series range
    
        '''
    
        
        
        # Convert start_date, end_date from local timezone to UTC.
        
        v_tz = pytz.timezone('America/Vancouver')
        
        #get meter data
        
        meter_data = []
        timestamp_data = None
        
        
        for meter in meters:

            cur = conn.cursor()
            
            current_time = datetime.datetime.now().strftime('%Y-%m-%d')

            cur.execute("SELECT tdss.timestamp AS timestamp, MAX(CASE When metric_key = 'kWh' THEN tdss.value END) AS kWh \
                                            FROM grid_element ge \
                                            JOIN grid_element_data_source geds \
                                                ON geds.grid_id = ge.grid_id \
                                                AND geds.grid_element_id = ge.grid_element_id \
                                            JOIN UNNEST(geds.metrics::TEXT[]) metric_key \
                                                ON true \
                                            JOIN ts_data_source_select(geds.grid_element_data_source_id, metric_key, %s) tdss \
                                                ON true \
                                            WHERE ge.grid_id = %s \
                                                AND ge.type = 'Meter' \
                                                AND ge.grid_element_id = %s \
                                                AND metric_key = 'kWh'\
                                            GROUP BY tdss.timestamp, geds.grid_element_id \
                                            ORDER BY 1, 2;", (f'[1800-01-01, {current_time}]', self.grid, meter))

            meter_data_sql = cur.fetchall()
            cur.close()

            conn.rollback()

            
            this_meter = pd.DataFrame(meter_data_sql, columns = ['timestamp', 'kwh'])
            
            if timestamp_data is None:
                timestamp_data = pd.DataFrame(this_meter.timestamp)
            
            #remove all columns except for timestamp and load, set index to timestamp
            
            this_meter = this_meter.rename(columns = {'kwh': ('load_' + meter)})
            
            
            meter_data.append(this_meter[['load_' + meter]])

            
 
        
        if timestamp_data is None:
            raise Exception('No time series data was found for elements downstream of this transformer.')
        
        # add up loads and index on time
        
        loads = timestamp_data.join(meter_data)
        loads['timestamp'] = loads['timestamp'].apply(lambda x: pd.Timestamp(x).astimezone(v_tz).tz_localize(None).asm8)  
        loads = loads.set_index('timestamp')
        
        loads = loads.sum(axis = 1)
            
        return loads


    def fit_ch_num_proportional(self, ch_prop_pairs):
        '''Determines the number of each charger that can fit in the given capacity, with the percentage of
            number of each type of charger used specified.

                    Parameters:
                            ch_prop_pairs: pairs of chargers and percentages (charger, %)
            

            The number of chargers is calculated by noting that if we have n chargers total, then our total
                power draw is (charger1_power*charger1_weight + ... + chargerk_power*chargerk_weight)*n
                which is equivalent to fitting one charger with power draw (charger1_power*charger1_weight + ... + chargerk_power*chargerk_weight).
            
        '''


        if type(ch_prop_pairs[-1]) is not tuple:
            total_per = np.sum([ch[1] for ch in ch_prop_pairs[:-1]])
            ch_prop_pairs[-1] = (ch_prop_pairs[-1], 1 - total_per)

        ch_prop_pairs = copy.deepcopy(ch_prop_pairs)
        ch_prop_pairs.sort(key = lambda y: y[0], reverse = True)


        agg_charger_power = np.sum([ch[0].power*ch[1] for ch in ch_prop_pairs if ch[0].voltage <= self.secondary_voltage])


        agg_charger = charger(power = agg_charger_power, voltage = 0)

        charger_data = self.fit_EVC(agg_charger)

        for i in range(0, len(ch_prop_pairs)):
            if ch_prop_pairs[i][0].voltage <= self.secondary_voltage:
                charger_data['Charger ' + str(i) + ': ' +  str(ch_prop_pairs[i][0].power) + 'kW'] = (charger_data['EVCs']*ch_prop_pairs[i][1]).apply(lambda x: floor(x))
            else:
                charger_data['Charger ' + str(i) + ': ' +  str(ch_prop_pairs[i][0].power) + 'kW'] = 0


        charger_data['Used Capacity'] = 0
        
        for i in range(0, len(ch_prop_pairs)):
            if ('Charger ' + str(i) + ': ' +  str(ch_prop_pairs[i][0].power) + 'kW' ) in charger_data.columns:
                charger_data['Used Capacity'] = charger_data['Used Capacity'] + \
                                                charger_data['Charger ' + str(i) + ': ' +  str(ch_prop_pairs[i][0].power) + 'kW']*ch_prop_pairs[i][0].power

        charger_data = charger_data.drop(['load', 'Excess Capacity', 'EVCs'], axis = 1)

        return charger_data.copy(deep = True)

    def fit_ch_pow_proportional(self, ch_prop_pairs):
        '''Determines the number of each charger that can fit in the given capacity, with the percentage of
            power of pulled by each type of charger used specified.

                    Parameters:
                            ch_prop_pairs: pairs of chargers and percentages (charger, %)
                                            percentage is the percentage of total budget
            
            
        '''

        if type(ch_prop_pairs[-1]) is not tuple:
            total_per = np.sum([ch[1] for ch in ch_prop_pairs[:-1]])
            ch_prop_pairs[-1] = (ch_prop_pairs[-1], 1 - total_per)


        ch_prop_pairs = copy.deepcopy(ch_prop_pairs)
        ch_prop_pairs.sort(key = lambda y: y[0], reverse = True)



        # Idea: split off a "pseudo-transformer" version of the current transformer for each charger, where the capacity budget of this pseudo-transformer is
        #   (charger proportion)*(budget of real transformer). Fit on each pseudo-transformer and then aggregate.

        charger_data = self.capacity_data.copy(deep = True)

        for i in range(0, len(ch_prop_pairs)):

            t = transformer(self.grid, self.id, self.power_factor, self.capacity_budget*ch_prop_pairs[i][1])
            
            t.capacity_data = self.capacity_data.copy(deep = True)
            t.capacity_data['Budgeted Capacity'] = t.capacity_data['Budgeted Capacity']*ch_prop_pairs[i][1]
            
            t.secondary_voltage = self.secondary_voltage
            t.rating_kva = self.rating_kva

            charger_data['Charger ' + str(i) + ': ' +  str(ch_prop_pairs[i][0].power) + 'kW'] = t.fit_EVC(ch_prop_pairs[i][0])['EVCs']

            del t

        for i in range(0, len(ch_prop_pairs)):
            if ch_prop_pairs[i][0].voltage > self.secondary_voltage:
                charger_data['Charger ' + str(i) + ': ' +  str(ch_prop_pairs[i][0].power) + 'kW'] = 0


        charger_data['Used Capacity'] = 0
        
        for i in range(0, len(ch_prop_pairs)):
            if ('Charger ' + str(i) + ': ' +  str(ch_prop_pairs[i][0].power) + 'kW' ) in charger_data.columns:
                charger_data['Used Capacity'] = charger_data['Used Capacity'] + \
                                                charger_data['Charger ' + str(i) + ': ' +  str(ch_prop_pairs[i][0].power) + 'kW']*ch_prop_pairs[i][0].power

        charger_data['Budgeted Capacity'] = charger_data['Excess Capacity']*self.capacity_budget
        charger_data = charger_data.drop(['load', 'Excess Capacity'], axis = 1)

        return charger_data.copy(deep = True)        


    

    def fit_full_onetime(excess_load, chargers, max_number):
        
        '''given a list of EV charger powers (in kWh), the function will return the 'near' optimal distribution
            of numbers of each charger type for the given excess load
                parameters: chargers(list): A list of chargers (power in kWh)
                            excess_load(float): excess load on a transformer
                            percent(float): percent of the load that we are allowed to assign to EV chargers
                returns: a tuple of int whose i-th entry is the number of chargers of i-th type. 
        '''
        chargers.sort(reverse = True)
        
        lower_bound = excess_load * 0.8       # atleast utilize 80% of the excess load
        set_ = [str(i) for i in range(ceil(excess_load/chargers[-1].power))]
        combinations = [tuples for tuples in itertools.product(set_, repeat = len(chargers))]
        func = lambda x: tuple(int(i) for i in x)
        combinations_int = tuple(func(i) for i in combinations)
        set_of_seq = []
        for x in combinations_int:
            if lower_bound <= np.sum(np.array(x) * np.array([ch.power for ch in chargers])) <= excess_load:
                set_of_seq.append(x)
        for i in range(len(set_of_seq)-1, -1, -1):
            if sum(set_of_seq[i]) > max_number:
                set_of_seq.remove(set_of_seq[i])
        prod_of_seq = np.array(set_of_seq)*np.array([ch.power for ch in chargers])
        sum_of_seq = []
        for i in range(len(prod_of_seq)):
            sum_of_seq.append(sum(prod_of_seq[i]))
        # return set_of_seq
        return set_of_seq[np.argmax(sum_of_seq)]
    
    def fit_EVC(self, ch):
        '''Determines the number of electric vehicle chargers, with the given parameters, fit within excess capacity
            on the specified transformer on an hourly basis.
            
            
        '''
        
        charger_power = ch.power

        charger_voltage = ch.voltage

        data = None
        
        if self.capacity_data is not None:
            data = self.capacity_data.copy(deep = True)
        else:
            raise Exception('No transformer data found. Call retrieve_data before fitting EVs to the transformer.')
        
        
        if self.secondary_voltage is None:
            raise Exception('No metadata found. Call retrieve_data before fitting EVs to the transformer.')

        if charger_voltage > self.secondary_voltage:
            raise Exception('Charger voltage is higher than transformer secondary voltage. Charger cannot be installed downstream of this transformer.')
        
        data['EVCs'] = (data['Budgeted Capacity']/charger_power).apply(lambda x: max(floor(x),0))
        
        return data.copy(deep = True)

    ### Graphing functions

    def graph_excess_capacity(self, start_date, end_date, 
                            bins = 40, 
                            aggregator = 'mean', 
                            ax = None,
                            figwidth = 12, figheight = 6):
        ''' Graphs the excess capacity for each transformer in a list of transformers.
            
                    Parameters:
                            start_date (String): start date of time series range
                            end_date (String): end date of time series range
                            bins (int): Number of intervals to graph. Data within each interval is averaged to
                                                give a single capacity number on that interval.
                            aggregator (String): if there are more timestamps than bins, specifies method to
                                                        aggregate on each bin. Options are 'mean', 'min'.
                                                            'mean': take the average excess capacity in each time interval
                                                            'min': take the minimum excess capacity in each time interval
                                                            'both': plot both average and minimum on the same axes
                            ax (matplotlib.axes): axis to graph on if a common plot is desired.
                            figwidth, figheight (float): if ax is not given, generate an axis with these dimensions.
        '''
        
        
        
        capacity_data = self.view_in_range(start_date, end_date)[['Budgeted Capacity']]

        range_min = capacity_data['Budgeted Capacity'].nsmallest(3)[-1]
        range_max = capacity_data['Budgeted Capacity'].nlargest(3)[-1]
        
        y_min = floor(range_min - (range_max - range_min)/3)
        y_max = ceil(range_max + (range_max - range_min)/3)
        
        start_dt = datetime.datetime.fromisoformat(start_date)
        end_dt = datetime.datetime.fromisoformat(end_date)
        
        if ax == None:
            fig, ax = plt.subplots(figsize = (figwidth, figheight))

        ax.set_ylim([y_min, y_max])
        
        if (end_dt - start_dt)/3600 < datetime.timedelta(hours = bins)/3600:
            
            sns.lineplot(data = capacity_data[['Budgeted Capacity']], ax = ax)
                
        else:
            
            loads_d = {}
            loads_d_min = {}
            loads_d_mean = {}
            
            for i in range(0, bins):
                sub_start = pd.Timestamp(start_dt + (i/bins)*(end_dt - start_dt)).asm8
                sub_end = pd.Timestamp(start_dt + ((i+1)/bins)*(end_dt - start_dt)).asm8
                
                #Making a choice to label each time interval with its end time.
                if aggregator == 'mean':
                    
                    loads_d[sub_end] = (capacity_data[(capacity_data.index > sub_start) 
                                                    & (capacity_data.index <= sub_end)]).mean()
                elif aggregator == 'min':
                    
                    loads_d[sub_end] = (capacity_data[(capacity_data.index > sub_start) 
                                                    & (capacity_data.index <= sub_end)]).min()
                elif aggregator == 'both':
                    
                    
                    loads_d_mean[sub_end] = (capacity_data[(capacity_data.index > sub_start) 
                                                    & (capacity_data.index <= sub_end)]).mean()
                    loads_d_min[sub_end] =  (capacity_data[(capacity_data.index > sub_start) 
                                                    & (capacity_data.index <= sub_end)]).min()
                    
                else:
                    raise Exception('aggregator not recognized. aggregator can be either "mean", "min", or "both".')
        
            if (aggregator == 'mean') or (aggregator == 'min'):
                loads_df = pd.DataFrame.from_dict(loads_d, orient = 'index')
            else:
                loads_df = pd.DataFrame.from_dict(loads_d_min, orient = 'index').add_suffix('_min')
                loads_df = loads_df.join(pd.DataFrame.from_dict(loads_d_mean, orient = 'index').add_suffix('_avg'))
            
            sns.lineplot(data = loads_df, ax = ax)
            
        
        title_mod = ""
        
        if aggregator == 'min':
            title_mod = 'Minimum '
            ax.legend().remove()
        elif aggregator == 'mean':
            title_mod = 'Average '
            ax.legend().remove()
        else:
            title_mod = 'Average and minimum '
            
            transformer_suffix = [self.id + ' minimum', self.id + ' average']
            plt.legend(labels = transformer_suffix, loc='upper left', bbox_to_anchor=(1,0.95))
        

        ax.xaxis.label.set_size(15)
        ax.yaxis.label.set_size(15)
        
        plt.title(title_mod + "available capacity on " + self.id, fontsize = 20)
        plt.xticks(rotation=45)
        plt.ylabel('Available Capacity (kVA)', fontsize = 17)
        plt.show()
        

    def graph_power_scenarios(self, ch_prop_pairs, ax = None, show_power = True, prop = 'num'):
        '''Graphs the charger distribution within the budgeted capacity for four scenarios:
            - chargers never surpass budget
            - chargers never surpass budget during the night
            - chargers never surpass budget during the summer
            - chargers never surpass budget during summer nights

            Chargers are given alongside a percentage. The percentage indicates what percentage of total
                number of chargers are of that type.
            
                    Parameters:
                            - ch_prop_pairs: list of tuples (charger, proportion). The proportion is a float between 0 and 1,
                                which indicates what percentage of the total chargers should be of the matching type.
                            - ax: matplotlib axis, only used if show_power = False
                            - show_power (bool): flag to determine if the capacity graph is shown alongside the charger distribution.   
                            - prop (str): determine the measure of the charger proportions
                                            - if prop = 'num', the chargers are fit to be proportional with the total number of chargers
                                            - if prop = 'pow', the chargers are fit to be proportional with the total power draw, where
                                                    the percentage given is treated as a maximum proportion of the budgeted capacity for that charger.
            
        '''

        if (prop != 'num') and (prop != 'pow'):
            raise Exception('Proportion measure, prop, must be either \'num\' or \'pow\'.')
        
        charger_data = None

        ch_prop_pairs = copy.deepcopy(ch_prop_pairs)
        if len(ch_prop_pairs) > 1:
            ch_prop_pairs.sort(key = lambda y: y[0], reverse = True)

        if prop == 'num':
            charger_data = self.fit_ch_num_proportional(ch_prop_pairs)        
        if prop == 'pow':
            charger_data = self.fit_ch_pow_proportional(ch_prop_pairs)    


        if charger_data is None:
            raise Exception('Error occured while fitting chargers to transformer: charger_data was returned as None.')    

        charger_data = transformer.parse_date_time(charger_data).drop(['year', 'day', 'weekday'], axis = 1)
        
        
        hourly_min = charger_data.drop('month', axis = 1).groupby(by = 'hour').min()
        summer_hourly_min = charger_data[(charger_data['month'] >= 6) & (charger_data['month'] <= 9)] \
                                        .drop('month', axis = 1) \
                                        .groupby(by = 'hour').min()
        
        
        charger_data['date'] = charger_data.index.date
        charger_data = charger_data.drop(['month'], axis = 1)
        
        
        total_min = hourly_min.min()
        nightly_min = hourly_min[(hourly_min.index <= 4) | (hourly_min.index >= 22)].min()
        summer_total_min = summer_hourly_min.min()
        summer_nightly_min = summer_hourly_min[(hourly_min.index <= 6) | (hourly_min.index >= 22)].min()
        

        #Setup dataframe for charger count graph
        

        df = pd.DataFrame(total_min.drop(['Budgeted Capacity', 'Used Capacity'], axis = 0).apply(lambda x: max(x,0.02)), columns = ['All time'])
        df['Nightly'] = nightly_min.drop(['Budgeted Capacity', 'Used Capacity'], axis = 0).apply(lambda x: max(x,0.02))
        df['All summer'] = summer_total_min.drop(['Budgeted Capacity', 'Used Capacity'], axis = 0).apply(lambda x: max(x,0.02))
        df['Summer nightly'] = summer_nightly_min.drop(['Budgeted Capacity', 'Used Capacity'], axis = 0).apply(lambda x: max(x,0.02 ))

        df = df.transpose()

        #Setup dataframe for power graph
        if show_power is True:
            budget_df = pd.DataFrame(total_min,
                                        columns = ['All time'])


            budget_df['Nightly'] = nightly_min
            budget_df['All summer'] = summer_total_min
            budget_df['Summer nightly'] = summer_nightly_min


            budget_df = budget_df.transpose()

            for col in budget_df.columns:
                if col[0:7] == 'Charger':
                    power = float(col.split()[2][:-2])
                
                    budget_df[col] = budget_df[col]*power
            
            budget_df['Leftover Available Capacity'] = budget_df['Budgeted Capacity'] - budget_df['Used Capacity']
            budget_df = budget_df.drop(['Budgeted Capacity', 'Used Capacity'], axis = 1)



        # Graphing

        if type(show_power) is not bool:
            raise Exception('show_power flag must be a boolean.')

        if show_power is True:
            fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (14,6))

            ax1 = axs.flatten()[0]
            ax2 = axs.flatten()[1]

        else: 
            if ax is None:
                fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (7,6))

                ax1 = axs
            else:
                ax1 = ax
        

        df.plot(kind = 'bar', stacked = True, ax = ax1)

        
        if (np.sum([total_min[i] for i in range(1,len(ch_prop_pairs)+1)]) < 3) and \
            (np.sum([nightly_min[i] for i in range(1,len(ch_prop_pairs)+1)]) < 3) and \
            (np.sum([summer_total_min[i] for i in range(1,len(ch_prop_pairs)+1)]) < 3) and \
            (np.sum([summer_nightly_min[i] for i in range(1,len(ch_prop_pairs)+1)]) < 3):
            ax.set_yticks(ticks = [0,1,2,3])

        for c in ax1.containers:

            # Optional: if the segment is small or 0, customize the labels
            labels = [int(v.get_height()) if v.get_height() > 0.5 else '' for v in c]
    
            # remove the labels parameter if it's not needed for customized labels
            ax1.bar_label(c, labels=labels, label_type='center')

        charger_powers = [str(ch_prop_pairs[i][0].power) + ' kW' for i in range(0,len(ch_prop_pairs))]

        ax1.legend(labels = charger_powers, loc='upper left', bbox_to_anchor=(1.02,0.95))
        ax1.tick_params('x', labelrotation= 45, size = 17)
        ax1.set_title(self.id, pad = 10, fontsize = 19)
        ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.set_ylabel('Number of chargers', fontsize = 17)

        if show_power is True:
            budget_df.plot(kind = 'bar', stacked = True, ax = ax2)

            for c in ax2.containers:

                # Optional: if the segment is small or 0, customize the labels
                labels = [int(v.get_height()) if v.get_height() > 0 else '' for v in c]
        
                # remove the labels parameter if it's not needed for customized labels
                ax2.bar_label(c, labels=labels, label_type='center')
            
                
            ax1.set_title('Number of chargers', fontsize = 19)
            ax2.legend(loc='upper left', bbox_to_anchor=(1.02,0.95))    
            ax2.tick_params('x', labelrotation= 45, size = 17)
            ax2.set_title('Power usage of chargers', fontsize = 19)
            ax2.set_ylabel('Power usage (kW)', fontsize = 17)

            fig.subplots_adjust(wspace = 0.6)

            fig.suptitle('Chargers on transformer ' + self.id, fontsize = 22, x = 0.52, y = 1.05)
        

        # Assume that if an axis is being passed, then graph parameters may still need to be adjusted before calling plt.show().

        if ax is None:
            if show_power is True:
                plt.figtext(0.145,-0.17,'Night is assumed to be between 10pm and 4am.\nSummer is assumed to be between June and September.', fontsize= 15)
            else:
                plt.figtext(0.17,-0.17,'Night is assumed to be between 10pm and 4am.\nSummer is assumed to be between June and September.', fontsize = 15)

            plt.show()

    def graph_capacity_with_chargers(self, ch_prop_pairs):
        
        charger_data = self.fit_ch_num_proportional(ch_prop_pairs)        

        charger_data = transformer.parse_date_time(charger_data).drop(['year', 'day', 'weekday'], axis = 1)
        
        
        hourly_min = charger_data.drop('month', axis = 1).groupby(by = 'hour').min()
        summer_hourly_min = charger_data[(charger_data['month'] >= 6) & (charger_data['month'] <= 9)] \
                                        .drop('month', axis = 1) \
                                        .groupby(by = 'hour').min()
        
        
        charger_data['date'] = charger_data.index.date
        charger_data = charger_data.drop(['month'], axis = 1)
        
        daily_min = charger_data.drop('hour', axis = 1).groupby(by = 'date').min()
        daily_min = daily_min.rename(columns = {'Budgeted Capacity': 'Minimum daily capacity'})
    
        daily_nightly_min = charger_data[(charger_data['hour'] >= 22) | (charger_data['hour'] <= 4)] \
                                        .drop('hour', axis = 1) \
                                        .groupby(by = 'date').min()
        daily_nightly_min = daily_nightly_min.rename(columns = {'Budgeted Capacity': 'Minimum nightly capacity'})

        total_min = hourly_min.min()['Used Capacity']
        nightly_min = hourly_min[(hourly_min.index <= 4) | (hourly_min.index >= 22)].min()['Used Capacity']
        summer_total_min = summer_hourly_min.min()['Used Capacity']
        summer_nightly_min = summer_hourly_min[(hourly_min.index <= 4) | (hourly_min.index >= 22)].min()['Used Capacity']

        charger_data = charger_data.drop(['hour', 'date', 'Budgeted Capacity'], axis = 1)

        charger_data['Used Capacity - Max any time'] = total_min
        charger_data['Used Capacity - Max at night'] = nightly_min
        charger_data['Used Capacity - Max at summer'] = summer_total_min
        charger_data['Used Capacity - Max at summer night'] = summer_nightly_min

        charger_data = charger_data.drop(['Used Capacity'], axis = 1)

        for col in charger_data.columns:
            if col.split()[0] == 'Charger':
                charger_data = charger_data.drop([col], axis = 1)
        

        fig, ax = plt.subplots(figsize = (14,8))
        ax.yaxis.get_major_locator().set_params(integer=True)

        sns.lineplot(data = daily_min[['Minimum daily capacity']], ax = ax, palette = 'Greens')
        sns.lineplot(data = daily_nightly_min[['Minimum nightly capacity']], ax = ax, palette = 'Blues')


        sns.lineplot(data = charger_data, ax = ax, palette = 'copper', dashes = [(1,0), (5,7) , (2, 4), (1,1)])

    

        plt.legend(loc='upper left', bbox_to_anchor=(1.05,0.95))
    
        plt.xticks(rotation = '45')
        plt.xlabel('Date')
        
        
        plt.ylabel('Capacity (kVA)')
        
        plt.title('Power available vs. different scenarios of charger consumption')
        
        plt.show()
        
    def graph_capacity_monthly(self):

        fig, ax = plt.subplots(figsize=(12, 6))

        
        data = self.capacity_data.copy(deep = True)

        data['month'] = data.index.month

        sns.boxplot(data = data, x = 'month', y = 'Budgeted Capacity', ax = ax, color = 'b')

        months = calendar.month_name[1:]
        ax.set_xticks(range(0, 12))
        ax.set_xticklabels(months, rotation=30, fontsize = 13)

        plt.xlabel("Month of the Year", fontsize = 17)
        plt.ylabel("Available capacity (kVA)", fontsize = 17)
        plt.suptitle('Available capacity on ' + ' '.join(self.id.split('_')) + ' by month', fontsize = 20)
        plt.show()

    def graph_capacity_hourly(self):

        fig, ax = plt.subplots(figsize=(12, 6))

        
        data = self.capacity_data.copy(deep = True)

        data['hour'] = data.index.hour

        sns.boxplot(data = data, x = 'hour', y = 'Budgeted Capacity', ax = ax, color = 'b')

        plt.xlabel("Hour of the day", fontsize = 17)
        plt.ylabel("Available capacity (kVA)", fontsize = 17)
        plt.suptitle('Available capacity on ' + ' '.join(self.id.split('_')) + ' by hour', fontsize = 20)
        plt.show()



    ### Helper functions


    def parse_date_time(df):
        '''Parses datetime information from timestamps in the index.'''
        
        df = df.copy(deep = True)
        
        days_d = dict(enumerate(calendar.day_name))
        
        df['weekday'] = df.index.dayofweek
        df['weekday'] = df['weekday'].apply(lambda x: days_d[x])
        
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['hour'] = df.index.hour
        
        return df

@total_ordering
class charger:

    def __init__(self, power, voltage):
        self.power = power
        self.voltage = voltage

    def __str__(self):
        return f'Charger: ({self.power}, {self.voltage})'
        
    def __repr__(self):
        return f'Charger: ({self.power}, {self.voltage})'


    def __eq__(self, __o: object) -> bool:
        
        return (self.power == __o.power) and (self.voltage == __o.voltage)

    def __lt__(self, __o: object) -> bool:


        return (self.power, self.voltage) < (__o.power, __o.voltage)


def graph_chargers_on(transformers, ch_prop_pairs):
    '''Takes a list of transformers and a list of budgets for each transformer and graphs the chargers
        that will fit on each transformer within the budget specified. Each graph contains four scenarios:
            - The chargers never pull more than the budgeted capacity.
            - The chargers never pull more than the budgeted capacity at night.
            - The chargers never pull more than the budgeted capacity during the summer.
            - The chargers never pull more than the budgeted capacity during summer nights.
                
                
                    Parameters:
                            - transformers (list of transformers): list of transformers to graph
                            - ch_prop_pairs: list of tuples (charger, proportion). The proportion is a float between 0 and 1,
                                which indicates what percentage of the total chargers should be of the matching type.
    '''
    
    n_cols = min(3, len(transformers))
    n_rows = ceil(len(transformers)/n_cols)

    fig, axs = plt.subplots(ncols = n_cols, nrows = n_rows, figsize = (1 + 5*n_cols, 4*n_rows + 0.2*(n_rows - 1)))

    if len(transformers) > 1:
        for i in range(len(transformers), len(axs.flatten())):
            axs.flatten()[i].set_axis_off()

    for i in range(0, len(transformers)):

        transformers[i].graph_power_scenarios(ch_prop_pairs, ax = axs.flatten()[i], show_power = False)

    fig.subplots_adjust(hspace = 0.7, wspace = 0.6)

    fig.suptitle('Chargers that fit on chosen transformers', x = 0.51, y = 1.04, fontsize = 22)
    plt.figtext(0.125,-0.3,'Night is assumed to be between 10pm and 4am.\nSummer is assumed to be between June and September.', fontsize = 20)
    plt.show()