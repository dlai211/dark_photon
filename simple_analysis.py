#!/usr/bin/env python
import ROOT
import time
import sys

# Start timer
start_time = time.time()

# Define the directory and file name
directory = "/data/tmathew/ntups/mc23d"
filename = "ggHyyd_y.root"
filepath = f"{directory}/{filename}"

print("Processing file:", filepath)

# Open the ROOT file
f = ROOT.TFile.Open(filepath)
if not f or f.IsZombie():
    print(f"Error: Could not open file {filepath}")
    sys.exit(1)

# Retrieve the TTree. Adjust the tree name if it is different.
tree = f.Get("nominal")
if not tree:
    print(f"Error: TTree 'nominal' not found in {filepath}")
    f.Close()
    sys.exit(1)

# Get the total number of events (raw, unweighted) before any cut
total_events = tree.GetEntries()
print(f"\nTotal events before any cut: {total_events}")

# Define the three separate cuts
cut1 = "met_tst_et > 50000"
cut2 = "trigger_HLT_g50_tight_xe40_cell_xe70_pfopufit_80mTAC_L1eEM26M"
cut3 = "ph_pt[0] > 50000"

# Apply each cut separately on the full TTree (not cumulatively)

# Cut 1
events_cut1 = tree.GetEntries(cut1)
print(f"\nAfter applying Cut 1: {cut1}")
print(f"  Events after Cut 1: {events_cut1}")

# Cut 2
events_cut2 = tree.GetEntries(cut2)
print(f"\nAfter applying Cut 2: {cut2}")
print(f"  Events after Cut 2: {events_cut2}")

# Cut 3
events_cut3 = tree.GetEntries(cut3)
print(f"\nAfter applying Cut 3: {cut3}")
print(f"  Events after Cut 3: {events_cut3}")

# Optionally, you can also show a cumulative cut flow.
# For example, apply all three cuts together:
cumulative_cut = f"{cut1} && {cut2} && {cut3}"
events_cumulative = tree.GetEntries(cumulative_cut)
print(f"\nAfter applying all three cuts cumulatively:")
print(f"  Events after all cuts: {events_cumulative}")

# Close the file and print processing time
f.Close()
print(f"\nTotal processing time: {time.time() - start_time:.2f} seconds")
