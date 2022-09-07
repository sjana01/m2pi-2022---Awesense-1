# m2pi-2022---Awesense

Awesense project from m2pi 2022: how many electric vehicle chargers can you fit on current grid infrastructure?

# Overview

This API works with Awesense's grid database to determine how many EV chargers fit on a transformer or list of transformers, using current and historical grid load. For a demo on how to use the API, see the [demo notebook.](https://github.com/thaddeusj/m2pi-2022---Awesense/blob/master/Awesense%20-%20Demo.ipynb)

Note that using this API requires an Awesense account with access to a grid database.

# Assumptions

This API assumes that data is stored in the database on a meter by meter database. After instantiating a transformer object, the transformer's data must be retrieved using ```transformer.retrieve_data```. This will populate the transformer's load DataFrame by adding up the power draws from each downstream meter of that distribution transformer or by adding up the power draws from each downstream transformer if the transformer is not a distribution transformer.

# Analysis

The API can analyse the EV charger capacity of a transformer in several different ways.

- Fit a single type of charger onto the transformer, by determining if the transformer outputs a high enough voltage for the charger and then dividing budgeted capacity by the charger's power draw if the voltage is high enough.
- Fit multiple types of chargers onto the transformer,
  - by assuming we want to use as much power from our available capactiy as possible,
  - by assuming we want a fixed proportion of each type of charger,
  - or by assuming we want a fixed proportion of the power draw of each type of charger.
  
The single charger type analysis is done via ```transformer.graph_chargers_on```, which can graph multiple transformers at the same time. This outputs a graph that considers availability at different times of day and year.

A similar analysis for the multiple charger type situation is performed by ```transformer.graph_power_scenarios```. This can handle any of our three constraints above, and gives a split for each charger type, as well as a summary of their power draws. These are also broken down by time, as in the single charger type case.


