import os
import matplotlib.pyplot as plt
import pandas as pd

def main():
    # Load data with missing values handled
    df = pd.read_csv('data/singularity.csv', quotechar='"', decimal=',', na_values=['', ' '])
    df.fillna(0, inplace=True)
    print(df.info())

    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 7))

    # Plot data
    axs[0].plot(df['Time'], df['Mechanical Units\\ROB_1 [IRB120_3_58__01]\\Joint\\Near Wrist Singularity'], 'b-')
    axs[0].set_title('Distance to singularity')
    axs[0].set_ylabel('Angle to singularity (deg)')
    axs[0].set_xlabel('Time (s)')
    axs[0].grid(True)

    # Calculate the derivative for each joint
    df['d1'] = (df['J1'] - df['J1'].shift(1)) / (df['Time'] - df['Time'].shift(1))
    df['d2'] = (df['J2'] - df['J2'].shift(1)) / (df['Time'] - df['Time'].shift(1))
    df['d3'] = (df['J3'] - df['J3'].shift(1)) / (df['Time'] - df['Time'].shift(1))
    df['d4'] = (df['J4'] - df['J4'].shift(1)) / (df['Time'] - df['Time'].shift(1))
    df['d5'] = (df['J5'] - df['J5'].shift(1)) / (df['Time'] - df['Time'].shift(1))
    df['d6'] = (df['J6'] - df['J6'].shift(1)) / (df['Time'] - df['Time'].shift(1))

    # Plot data
    axs[1].plot(df['Time'], df['d4'], 'r', label='Joint 4')
    axs[1].plot(df['Time'], df['d6'], 'g--', label='Joint 6')
    axs[1].set_title('Joint velocity')
    axs[1].set_ylabel('Velocity (deg/s)')
    axs[1].set_xlabel('Time (s)')
    axs[1].legend()
    axs[1].grid(True)

    # Plot movement speed
    axs[2].plot(df['Time'], df['Mechanical Units\\ROB_1 [IRB120_3_58__01]\\TCP\\Speed In Current Wobj'], 'm-')
    axs[2].set_title('Movement speed in world coordinates')
    axs[2].set_ylabel('Speed (mm/s)')
    axs[2].set_xlabel('Time (s)')
    axs[2].grid(True)

    # Adjust layout
    plt.subplots_adjust(hspace=0.75)
    fig.align_ylabels(axs)

    plt.show()

if __name__ == '__main__':
    main()