import pandas as pd
import pm4py
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.visualization.dfg import visualizer as dfg_visualizer
# from pm4py.visualization.common.gview import get_layout # For layout - Old import
import pydotplus # For parsing DOT output with layout
import re # For parsing edge spline points
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import os
import numpy as np # For random offsets

def load_and_prepare_log(csv_path):
    """Loads event log from CSV, renames columns, converts types, sorts, and augments with START/END nodes."""
    try:
        event_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Log file not found at {csv_path}")
        return None, None

    # Define standard column names for pm4py
    column_mapping = {
        'case_id': 'case:concept:name',
        'activity': 'concept:name',
        'activity_start_dt': 'time:timestamp'
    }
    
    # Check if necessary columns exist
    missing_cols = [col for col in column_mapping.keys() if col not in event_df.columns]
    if missing_cols:
        print(f"Error: Missing required columns in CSV: {', '.join(missing_cols)}")
        print(f"Expected columns: {', '.join(column_mapping.keys())}")
        return None, None

    event_df = event_df.rename(columns=column_mapping)

    # Convert timestamp and sort
    original_timestamps = event_df['time:timestamp'].copy() # Keep a copy for error reporting
    try:
        # Attempt 1: Parse with dayfirst=True (expected format)
        event_df['time:timestamp'] = pd.to_datetime(event_df['time:timestamp'], dayfirst=True, errors='coerce')
        
        # Identify rows that failed the first attempt
        failed_first_attempt_mask = event_df['time:timestamp'].isnull()
        if failed_first_attempt_mask.any():
            print(f"Info: {failed_first_attempt_mask.sum()} timestamps failed initial parsing (dayfirst=True). Attempting fallback (dayfirst=False).")
            # Attempt 2: Parse with dayfirst=False for those that failed
            # Only apply to rows that failed, using original_timestamps for those rows
            failed_timestamps_original = original_timestamps[failed_first_attempt_mask]
            parsed_fallback = pd.to_datetime(failed_timestamps_original, dayfirst=False, errors='coerce')
            event_df.loc[failed_first_attempt_mask, 'time:timestamp'] = parsed_fallback
            
            still_failed_mask = event_df['time:timestamp'].isnull()
            if still_failed_mask.any():
                print(f"Warning: {still_failed_mask.sum()} timestamps still failed parsing after fallback (dayfirst=False).")

    except Exception as e:
        print(f"Error during 'time:timestamp' conversion: {e}")
        print("Please ensure 'activity_start_dt' is in a recognizable datetime format (e.g., dd/mm/yyyy HH:MM:SS or mm/dd/yyyy HH:MM:SS).")
        return None, None

    # Check for any remaining NaT values and report/drop them
    rows_with_parsing_errors = event_df['time:timestamp'].isnull()
    if rows_with_parsing_errors.any():
        num_errors = rows_with_parsing_errors.sum()
        print(f"Error: {num_errors} timestamps could not be parsed and will be dropped.")
        print("Original values of 'activity_start_dt' for rows with parsing errors:")
        print(original_timestamps[rows_with_parsing_errors])
        event_df = event_df.dropna(subset=['time:timestamp'])
        if event_df.empty:
            print("Error: All rows were dropped due to timestamp parsing errors. Cannot proceed.")
            return None, None
        print(f"Info: Proceeding with {len(event_df)} rows after dropping unparseable timestamps.")
        
    # Original sort by time, then we will augment and re-sort by case & time
    event_df = event_df.sort_values('time:timestamp').reset_index(drop=True)

    # Augment log with START_NODE and END_NODE events for each case
    print("Info: Augmenting event log with START_NODE and END_NODE for each case...")
    augmented_events_list = []
    # Ensure all relevant columns from original_df are present for new START/END events
    # If other columns are needed by pm4py or for consistency, they should be included here
    # For now, only core pm4py columns and case_id are strictly necessary for START/END from original data
    
    # Create a minimal set of columns to copy for START/END nodes to avoid issues with non-numeric data
    # if other columns from the original log were causing issues during pd.concat
    cols_to_copy_for_synthetic_events = ['case:concept:name'] 
    # Add any other columns that might be needed and are safe to copy (e.g., numeric, string but not complex objects)

    for case_id, group in event_df.groupby('case:concept:name'):
        if group.empty:
            continue
        group = group.sort_values('time:timestamp')
        first_event_original = group.iloc[0]
        last_event_original = group.iloc[-1]

        start_event_data = {col: first_event_original[col] for col in cols_to_copy_for_synthetic_events}
        start_event_data['concept:name'] = 'START_NODE'
        start_event_data['time:timestamp'] = first_event_original['time:timestamp'] - pd.Timedelta(seconds=1)
        augmented_events_list.append(start_event_data)

        # Add all original events for the case
        augmented_events_list.extend(group.to_dict('records'))

        end_event_data = {col: last_event_original[col] for col in cols_to_copy_for_synthetic_events}
        end_event_data['concept:name'] = 'END_NODE'
        end_event_data['time:timestamp'] = last_event_original['time:timestamp'] + pd.Timedelta(seconds=1)
        augmented_events_list.append(end_event_data)

    if augmented_events_list:
        # event_df = pd.concat([event_df, pd.DataFrame(augmented_events_list)], ignore_index=True) # This was adding duplicates
        event_df = pd.DataFrame(augmented_events_list) # Rebuild with augmented events
        event_df = event_df.sort_values(['case:concept:name', 'time:timestamp']).reset_index(drop=True)
    
    # Filter to necessary columns for pm4py
    event_df_pm = event_df[['case:concept:name', 'concept:name', 'time:timestamp']].copy()
    
    # Convert to pm4py event log
    try:
        log = log_converter.apply(event_df_pm, parameters={log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'case:concept:name'})
    except Exception as e:
        print(f"Error converting DataFrame to pm4py log object: {e}")
        return None, None
        
    return log, event_df # Return original df for full event data

def parse_node_pos(pos_str):
    """Parses node 'pos' string like 'x,y' into (float(x), float(y))."""
    parts = pos_str.strip('\"').split(',')
    return float(parts[0]), float(parts[1])

def parse_edge_pos(pos_str):
    """Parses edge 'pos' string (spline) like 'e,x,y x1,y1 x2,y2...' into list of points."""
    points = []
    # Format can be "e,endx,endy x1,y1 x2,y2 ... startx,starty" or just "x1,y1 x2,y2 ..."
    # We are interested in the spline points. The format often includes start/end arrow indicators.
    # Example: "e,162.09,308.94 144.09,308.94 117.06,308.94 96.74,301.5"
    # The 'e,' part indicates the end point with an arrow. Other parts are spline control points.
    # We can try to extract all coordinate pairs.
    
    # Remove "s,..." and "e,..." markers if present, and split by space
    pos_str = re.sub(r'[se],([\d\.]+),([\d\.]+)\s*', '', pos_str.strip('\"'))
    coord_pairs = pos_str.split(' ')
    for pair in coord_pairs:
        if ',' in pair:
            try:
                x, y = map(float, pair.split(','))
                points.append((x, y))
            except ValueError:
                print(f"Warning: Could not parse coordinate pair '{pair}' in edge position string '{pos_str}'")
    return points

def get_process_model_and_layout(log):
    """Discovers DFG and computes layout using pydotplus to parse Graphviz output."""
    if log is None:
        return None, None
    
    layout_dict = {}
    dfg = None
    try:
        dfg = dfg_discovery.apply(log)
        
        # Get the graphviz.Digraph object from pm4py
        gviz_object = dfg_visualizer.apply(dfg, log=log, variant=dfg_visualizer.Variants.FREQUENCY, parameters={"format": "dot"})
        
        if gviz_object is None:
            print("Error: pm4py's DFG visualizer did not return a Graphviz object.")
            return dfg, None # Return DFG for potential fallback, but no layout

        # Use pipe() to get DOT string with layout attributes from Graphviz executable
        # This requires Graphviz 'dot' executable to be in PATH
        print("Info: Piping DFG to Graphviz 'dot' command for layout...")
        dot_with_layout_str = gviz_object.pipe(format='dot').decode('utf-8')
        
        if not dot_with_layout_str:
            print("Error: Piping to 'dot' returned empty string. Check Graphviz installation and PATH.")
            return dfg, None

        print("Info: Parsing DOT string with layout using pydotplus...")
        graphs = pydotplus.graph_from_dot_data(dot_with_layout_str)
        if not graphs:
            print("Error: pydotplus could not parse the DOT string.")
            return dfg, None
        
        graph = graphs[0] # Assuming a single graph

        # Extract node positions
        for node in graph.get_nodes():
            node_name = node.get_name().strip('\"')
            attrs = node.get_attributes()
            if 'pos' in attrs:
                try:
                    pos = parse_node_pos(attrs['pos'])
                    if node_name not in layout_dict: layout_dict[node_name] = {}
                    layout_dict[node_name]['pos'] = pos
                except Exception as e_node_pos:
                    print(f"Warning: Could not parse 'pos' for node {node_name}: {attrs.get('pos')}. Error: {e_node_pos}")
            # else:
            # print(f"Debug: Node {node_name} has no 'pos' attribute. Attrs: {attrs}")


        # Extract edge positions (splines)
        for edge in graph.get_edges():
            source_name = edge.get_source().strip('\"')
            target_name = edge.get_destination().strip('\"')
            attrs = edge.get_attributes()
            if 'pos' in attrs:
                try:
                    spline_points = parse_edge_pos(attrs['pos'])
                    if spline_points:
                         # Ensure source and target nodes are in layout_dict from node processing
                        if source_name in layout_dict and target_name in layout_dict:
                            layout_dict[(source_name, target_name)] = {'pos': spline_points}
                        # else:
                            # print(f"Debug: Source '{source_name}' or target '{target_name}' of edge not found in node positions during edge processing.")
                except Exception as e_edge_pos:
                    print(f"Warning: Could not parse 'pos' for edge {source_name}->{target_name}: {attrs.get('pos')}. Error: {e_edge_pos}")
            # else:
            # print(f"Debug: Edge {source_name}->{target_name} has no 'pos' attribute. Attrs: {attrs}")

        if not layout_dict or all('pos' not in val for val in layout_dict.values() if isinstance(val, dict)):
             print("Warning: Layout dictionary is empty or contains no position data after parsing. Animation might be incorrect.")
             print("This could happen if Graphviz 'dot' did not add 'pos' attributes, or parsing failed.")
             # Fallback to basic layout if parsing failed substantially
             raise ValueError("Layout parsing failed to yield positions.")

        return dfg, layout_dict

    except ImportError:
        print("Error: pydotplus library not found. Please install it: pip install pydotplus")
        return None, None # DFG might be available, but layout failed.
    except Exception as e:
        print(f"Error during DFG discovery or layout generation (pydotplus path): {e}")
        print("This might be due to Graphviz not being installed/in PATH, or an issue with pydotplus parsing.")
        print("Attempting a very basic fallback layout.")
        
        # Attempt to get DFG if not already available
        if 'dfg' not in locals():
            try:
                dfg = dfg_discovery.apply(log)
            except Exception as e_dfg_fallback:
                print(f"Critical: Could not even discover DFG for fallback: {e_dfg_fallback}")
                return None, None
        
        if dfg is None: 
            print("Critical: DFG is None, cannot proceed with fallback layout.")
            return None, None

        # Basic fallback layout (less ideal, as used before)
        dfg_nodes = set()
        if isinstance(dfg, dict): # DFG is typically a dict of (source, target) -> freq
            for src, tgt in dfg.keys():
                dfg_nodes.add(src)
                dfg_nodes.add(tgt)
        
        activity_names = sorted(list(dfg_nodes))
        if not activity_names: # If DFG was empty or nodes couldn't be extracted
            # Try to get activities from the log as a last resort for fallback
            activity_names = sorted(list(set(trace["concept:name"] for case in log for trace in case)))

        activity_to_pos_fallback = {act: (i*100, (i%3)*100 + np.random.randint(-10,10)) for i, act in enumerate(activity_names)}
        
        fallback_layout = {}
        for act_name, pos_val in activity_to_pos_fallback.items():
            fallback_layout[act_name] = {'pos': pos_val}
        
        print("Warning: Using a very basic random layout due to previous errors.")
        return dfg, fallback_layout

# New Helper Function
def interpolate_path(path_points, fraction):
    """Interpolates a point along a path defined by a list of (x,y) coordinates."""
    if not path_points: return None
    if len(path_points) == 1: return path_points[0]
    
    # Ensure fraction is within [0, 1]
    fraction = max(0.0, min(1.0, fraction))

    target_idx_float = fraction * (len(path_points) - 1)
    idx0 = int(target_idx_float)
    
    # If fraction is 1.0, target_idx_float can be len(path_points) - 1, making idx0 the last index.
    # In this case, idx1 should also be the last index.
    if idx0 >= len(path_points) - 1:
        return path_points[-1]
        
    idx1 = idx0 + 1 # min(idx0 + 1, len(path_points) - 1) is implicitly handled by idx0 check

    interp_frac_segment = target_idx_float - idx0
    
    p0 = path_points[idx0]
    p1 = path_points[idx1]
    
    interp_x = p0[0] + interp_frac_segment * (p1[0] - p0[0])
    interp_y = p0[1] + interp_frac_segment * (p1[1] - p0[1])
    return (interp_x, interp_y)

def create_animation(log_df, dfg, layout, output_path, max_days_to_animate=None):
    """Creates and saves the process animation."""
    if log_df is None or dfg is None or layout is None:
        print("Skipping animation due to missing log, DFG, or layout.")
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    token_radius = 0.03 # Adjusted for potentially larger coordinate spaces from graphviz
    
    # Extract node positions from the layout
    # Layout from get_graph_layout has nodes as keys, with 'pos' being (x,y)
    activity_to_pos = {node_name: data['pos'] for node_name, data in layout.items() if 'pos' in data and isinstance(node_name, str)}

    if not activity_to_pos:
        print("Error: Could not determine node positions from layout for animation. Cannot proceed.")
        return # Critical error

    # --- Animation Framing Logic for 3-hour intervals ---
    if log_df.empty or 'time:timestamp' not in log_df.columns or log_df['time:timestamp'].isnull().all():
        print("Error: Log DataFrame is empty or 'time:timestamp' column is missing/empty. Cannot create animation.")
        return

    min_log_time_precise = log_df['time:timestamp'].min() 
    max_log_time_precise = log_df['time:timestamp'].max()
    
    if pd.isna(min_log_time_precise) or pd.isna(max_log_time_precise):
        print("Error: Could not determine precise min/max log time for animation. Timestamps might be invalid.")
        return
        
    total_duration_seconds = (max_log_time_precise - min_log_time_precise).total_seconds()
    interval_hours = 3
    num_total_intervals_in_log = int(np.ceil(total_duration_seconds / (interval_hours * 3600))) + 1

    num_animation_intervals = num_total_intervals_in_log
    effective_max_time = max_log_time_precise

    if max_days_to_animate is not None and max_days_to_animate > 0:
        print(f"Info: User requested to limit animation to {max_days_to_animate} days.")
        # Calculate intervals for the user-specified day limit, starting from min_log_time_precise
        limit_duration_seconds = max_days_to_animate * 24 * 3600
        user_requested_intervals = int(np.ceil(limit_duration_seconds / (interval_hours * 3600)))
        
        # Ensure we don't exceed the actual log content
        num_animation_intervals = min(num_total_intervals_in_log, user_requested_intervals)
        # Calculate the effective end time for the limited animation
        effective_max_time = min_log_time_precise + pd.Timedelta(seconds=(num_animation_intervals -1) * interval_hours * 3600)
        if effective_max_time > max_log_time_precise:
            effective_max_time = max_log_time_precise # Cap at actual max log time if user days exceed it
        
        if num_animation_intervals < 1: # Handle cases where requested days is too short
            print("Warning: Requested day limit is too short for any animation intervals. Defaulting to 1 interval if possible.")
            num_animation_intervals = 1 if num_total_intervals_in_log > 0 else 0

    if num_animation_intervals == 0:
        print("Error: No animation intervals to render based on log data and day limit.")
        return

    print(f"Info: Animation will run for {num_animation_intervals} ({interval_hours}-hour) intervals.")
    print(f"Covering time from {min_log_time_precise.strftime('%Y-%m-%d %H:%M:%S')} to {effective_max_time.strftime('%Y-%m-%d %H:%M:%S')}.")
    # --- End Animation Framing Logic ---

    # Determine axis limits from layout
    if activity_to_pos:
        all_x = [pos[0] for pos in activity_to_pos.values()]
        all_y = [pos[1] for pos in activity_to_pos.values()]
        if all_x and all_y:
            ax.set_xlim(min(all_x) - 50, max(all_x) + 50)
            ax.set_ylim(min(all_y) - 50, max(all_y) + 50)
        else:
            ax.set_xlim(0, len(activity_to_pos)*100 if activity_to_pos else 500) # Fallback limits
            ax.set_ylim(-50, 250 if activity_to_pos else 200) # Fallback limits

    token_artists = {} # Stores {case_id: matplotlib_artist_for_token}

    def draw_static_elements():
        ax.clear()
        if activity_to_pos:
            all_x = [pos[0] for pos in activity_to_pos.values()]
            all_y = [pos[1] for pos in activity_to_pos.values()]
            if all_x and all_y: # Ensure lists are not empty
                 ax.set_xlim(min(all_x) - 50, max(all_x) + 50)
                 ax.set_ylim(min(all_y) - 50, max(all_y) + 50)
            else: # Fallback if all_x or all_y are empty
                ax.set_xlim(0, 500)
                ax.set_ylim(0, 200)


        # Draw edges (from DFG and layout)
        for (source, target), freq in dfg.items():
            if source in activity_to_pos and target in activity_to_pos:
                pos_source = activity_to_pos[source]
                pos_target = activity_to_pos[target]
                
                # Check if edge path points are in layout (for curved edges from Graphviz)
                edge_key = (source, target)
                if layout.get(edge_key) and 'pos' in layout[edge_key] and isinstance(layout[edge_key]['pos'], list):
                    edge_points = np.array(layout[edge_key]['pos'])
                    ax.plot(edge_points[:,0], edge_points[:,1], 'gray', lw=0.5 + freq*0.05, zorder=1)
                else: # Straight line
                    ax.plot([pos_source[0], pos_target[0]],
                            [pos_source[1], pos_target[1]],
                            'gray', lw=0.5 + freq*0.05, zorder=1)

        # Draw nodes
        for activity, pos in activity_to_pos.items():
            node_color = 'skyblue'
            text_color = 'black'
            if activity == 'START_NODE' or activity == 'END_NODE':
                node_color = 'green'
                text_color = 'black'
            
            ax.scatter(pos[0], pos[1], s=1000, c=node_color, zorder=2, ec='black', alpha=0.8)
            ax.text(pos[0], pos[1], activity, ha='center', va='center', fontsize=8, zorder=3, color=text_color)
        
        ax.set_xticks([])
        ax.set_yticks([])
        return list(ax.patches) + list(ax.lines) + list(ax.texts)


    def update_animation(frame_num):
        # Clear previous tokens (artists)
        for artist_list in token_artists.values(): # token_artists stores lists of artists per case
            for artist in artist_list:
                artist.remove()
        token_artists.clear()

        # --- Updated for 3-Hour Interval Frames ---
        current_interval_start_time = min_log_time_precise + pd.Timedelta(hours=frame_num * interval_hours)
        current_interval_end_time = current_interval_start_time + pd.Timedelta(hours=interval_hours) - pd.Timedelta(microseconds=1)
        ax.set_title(f"Process Animation - Interval {frame_num + 1}/{num_animation_intervals} - Time: {current_interval_start_time.strftime('%Y-%m-%d %H:%M:%S')}", fontsize=10)
        # --- End Update for 3-Hour Interval Frames ---

        active_tokens_info_this_frame = [] 
        
        for case_id, group in log_df.groupby('case:concept:name'):
            group = group.sort_values('time:timestamp').reset_index(drop=True)
            
            # Find the latest event_i such that event_i.time <= current_interval_start_time
            event_i = None
            event_i_idx = -1
            for idx, row in group.iterrows():
                if row['time:timestamp'] <= current_interval_start_time:
                    event_i = row
                    event_i_idx = idx
                else:
                    break # Events are sorted, so we can stop early
            
            if event_i is None: # Case hasn't started by current_interval_start_time
                continue

            event_i_activity_name = event_i['concept:name']
            pos_at_event_i = activity_to_pos.get(event_i_activity_name)
            if not pos_at_event_i: continue

            # Determine if the case is at event_i's node or transitioning
            if event_i_idx + 1 < len(group):
                event_i_plus_1 = group.iloc[event_i_idx + 1]
                event_i_plus_1_activity_name = event_i_plus_1['concept:name']
                pos_at_event_i_plus_1 = activity_to_pos.get(event_i_plus_1_activity_name)

                if not pos_at_event_i_plus_1: # Should not happen if layout is complete
                     active_tokens_info_this_frame.append((case_id, pos_at_event_i, True, event_i_activity_name))
                     continue

                # If current_interval_start_time is AT or AFTER the next event, token should be AT next event (or further)
                # This is handled by the outer loop logic finding the correct event_i already.
                # The current event_i is the one active *during* or *at the start of* current_interval_start_time

                if current_interval_start_time < event_i_plus_1['time:timestamp']:
                    # Case is between event_i and event_i_plus_1
                    total_segment_duration_seconds = (event_i_plus_1['time:timestamp'] - event_i['time:timestamp']).total_seconds()
                    time_into_segment_seconds = (current_interval_start_time - event_i['time:timestamp']).total_seconds()
                    
                    progress_fraction = 0.0
                    if total_segment_duration_seconds > 0:
                        progress_fraction = time_into_segment_seconds / total_segment_duration_seconds
                    progress_fraction = max(0.0, min(1.0, progress_fraction)) # Clamp to [0,1]

                    edge_path_points = layout.get((event_i_activity_name, event_i_plus_1_activity_name), {}).get('pos')
                    interpolated_pos = None
                    if edge_path_points and len(edge_path_points) > 0:
                        interpolated_pos = interpolate_path(edge_path_points, progress_fraction)
                    else: # Straight line fallback if no specific path or only one point
                        interpolated_pos = (
                            pos_at_event_i[0] + progress_fraction * (pos_at_event_i_plus_1[0] - pos_at_event_i[0]),
                            pos_at_event_i[1] + progress_fraction * (pos_at_event_i_plus_1[1] - pos_at_event_i[1])
                        )
                    
                    if interpolated_pos:
                        # If progress is effectively 0, it means current_interval_start_time is at event_i's time.
                        # If progress is effectively 1, it means current_interval_start_time is at event_i_plus_1's time.
                        # The logic should place it on the node if progress_fraction is ~0 or ~1 and current time matches event time.
                        # For simplicity now: if time_into_segment is 0, it's at node. Otherwise, transitioning.
                        if time_into_segment_seconds == 0 and event_i['time:timestamp'] == current_interval_start_time:
                             active_tokens_info_this_frame.append((case_id, pos_at_event_i, True, event_i_activity_name))
                        else: # Transitioning
                             active_tokens_info_this_frame.append((case_id, interpolated_pos, False, None))
                    else: # Fallback if interpolation somehow fails
                         active_tokens_info_this_frame.append((case_id, pos_at_event_i, True, event_i_activity_name))
                else: # current_interval_start_time is >= event_i_plus_1's time, so it's at event_i_plus_1 (or later)
                    # This case should have been handled by the initial loop finding the correct event_i.
                    # If we reach here, it means event_i is the correct one and it's effectively at the node.
                    active_tokens_info_this_frame.append((case_id, pos_at_event_i, True, event_i_activity_name))
            else: # This is the last event for the case (e.g., END_NODE)
                active_tokens_info_this_frame.append((case_id, pos_at_event_i, True, event_i_activity_name))
        
        # --- Drawing tokens based on active_tokens_info_this_frame ---
        current_artists_collection = []
        drawn_node_positions_this_frame = {} # For offsetting tokens on the SAME activity node

        for c_id, token_pos_tuple, is_on_node, activity_name_for_node_offset in active_tokens_info_this_frame:
            final_x, final_y = token_pos_tuple
            if is_on_node:
                # Draw a circle at the node position
                token = plt.Circle((final_x, final_y), token_radius, color='red', zorder=10, alpha=0.9)
                ax.add_artist(token)
                case_text = ax.text(final_x, final_y, str(c_id),
                                    ha='center', va='center', fontsize=6, color='white', zorder=11,
                                    bbox=dict(boxstyle="circle,pad=0.1", fc="red", ec="red", alpha=0.9))
                token_artists[c_id] = [token, case_text]
                current_artists_collection.extend([token, case_text])
            else:
                # Draw a line from the node to the interpolated position
                ax.plot([final_x, final_x], [final_y, final_y], 'red', lw=0.5, zorder=10)
                case_text = ax.text(final_x, final_y, str(c_id),
                                    ha='center', va='center', fontsize=6, color='white', zorder=11,
                                    bbox=dict(boxstyle="circle,pad=0.1", fc="red", ec="red", alpha=0.9))
                token_artists[c_id] = [case_text]
                current_artists_collection.append(case_text)

        return current_artists_collection


    print("Initializing animation elements...")
    init_artists = draw_static_elements() # Draw initial DFG
    
    print(f"Creating animation with {num_animation_intervals} intervals...")
    # blit=True can be faster but sometimes causes issues with dynamic text/artists
    # If issues, try blit=False
    #Interval can be increased for slower animation
    ani = FuncAnimation(fig, update_animation, frames=num_animation_intervals, 
                        init_func=lambda: init_artists, blit=False, repeat=False, interval=200) # 200ms interval

    # Save animation
    animation_filename = os.path.join(os.path.dirname(output_path), "process_animation.gif")
    try:
        print(f"Attempting to save animation to {animation_filename} using imagemagick...")
        ani.save(animation_filename, writer='imagemagick', fps=5) # fps=5 for 200ms interval
        print(f"Animation saved successfully: {animation_filename}")
    except Exception as e_gif:
        print(f"Failed to save GIF with imagemagick: {e_gif}")
        print("Make sure ImageMagick is installed and in your system's PATH.")
        animation_filename_mp4 = os.path.join(os.path.dirname(output_path), "process_animation.mp4")
        try:
            print(f"Attempting to save animation to {animation_filename_mp4} using ffmpeg...")
            ani.save(animation_filename_mp4, writer='ffmpeg', fps=5, dpi=100) # fps=5
            print(f"Animation saved successfully as MP4: {animation_filename_mp4}")
        except Exception as e_mp4:
            print(f"Failed to save MP4 with ffmpeg: {e_mp4}")
            print("Make sure FFmpeg is installed and in your system's PATH.")
            print("Animation not saved. You can try plt.show() to display it if running in an interactive environment.")
            # plt.show() # Uncomment to display if saving fails and in interactive env

def main():
    parser = argparse.ArgumentParser(description="Create a process animation from an event log CSV.")
    parser.add_argument("--log", required=True, help="Path to the event log CSV file.")
    parser.add_argument("--days", type=int, default=None, help="Optional: Limit animation to the first N days of the event log. If not set, full log duration is used.")
    args = parser.parse_args()

    print(f"Processing event log: {args.log}")
    pm4py_log, full_log_df = load_and_prepare_log(args.log)

    if pm4py_log is None or full_log_df is None:
        print("Exiting due to errors in log preparation.")
        return

    print("Discovering process model and layout...")
    dfg, layout = get_process_model_and_layout(pm4py_log)
    
    if dfg is None or layout is None:
        print("Exiting due to errors in DFG/layout generation.")
        return

    create_animation(full_log_df, dfg, layout, args.log, args.days)
    print("Script finished.")

if __name__ == "__main__":
    main() 