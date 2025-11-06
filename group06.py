# ==================== IMPORTS ====================
import argparse
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm
from sqlalchemy import create_engine
from contextlib import contextmanager
from typing import List, Tuple, Optional, Dict
import random

# ==================== CONSTANTS ====================

# DATABASE CONFIGURATION
# Uses environment variables with fallback defaults for security and flexibility
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),  # Database server address
    "port": int(os.getenv("DB_PORT", "3306")),  # MariaDB default port
    "user": os.getenv("DB_USER", "coen2220"),  # Database username
    "password": os.getenv("DB_PASSWORD", "coen2220"),  # Database password
    "database": os.getenv("DB_NAME", "group06"),  # Target database name
}

# TEMPORAL DATA STRUCTURES
# Full month names used for display and data validation
MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

# Abbreviated month names for compact x-axis labels in plots
MONTH_ABBREV = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Dictionary mapping month names to numbers (1-12) for sorting and numerical operations
# This enables conversion from categorical month names to ordinal values
MONTH_MAP = {month: i + 1 for i, month in enumerate(MONTHS)}

# COLOR SCHEME FOR TEMPERATURE VISUALIZATION
TEMP_COLORS = ["#1f628c",  # Deep blue
               "#64aaaf",  # Teal
               "#a2deaf",  # Light cyan
               "#c5e7ad",  # Light green
               "#fff4c6",  # Cream
               "#fdc87f",  # Peach
               "#f89c56",  # Orange
               "#e65a3a",  # Red-orange
               "#a11216"]  # Deep red

# DARK THEME CONFIGURATION
# Modern dark mode aesthetic for graphs 6-10
DARK_THEME = {
    "bg": "#0b0c10",  # Very dark blue-black for outer background
    "plot_bg": "#11141a",  # Slightly lighter for plot area
    "text": "white"
}

# SEASON GROUPINGS
# Maps months to meteorological seasons for seasonal analysis (graph 13)
SEASONS = {
    "Winter": ["December", "January", "February"],
    "Spring": ["March", "April", "May"],
    "Summer": ["June", "July", "August"],
    "Autumn": ["September", "October", "November"]
}

# VISUALIZATION PARAMETERS
FIGURE_SIZE = (12, 6)
DPI = 100

# DATA PROCESSING PARAMETERS
BATCH_SIZE = 5000  # Number of rows to insert at once
TOP_AREAS_COUNT = 30  # How many top regions to display

# ANALYSIS THRESHOLDS
TEMP_THRESHOLDS = [1.5, 2.0]  # Paris Agreement temperatures (°C)
DECADE_YEARS = [1961, 1970, 1980, 1990, 2000, 2010, 2019]  # Years for decade analysis

# SECURITY: SQL INJECTION PREVENTION
ALLOWED_SOURCE_TABLES = ["temperature_data", "climate_data", "temp_raw"]


# ==================== DATABASE ====================
class DatabaseManager:

    def __init__(self, config: dict):
        #Initialize database manager with configuration
        self.config = config
        # Create SQLAlchemy engine on initialization for connection pooling
        # Engine is thread-safe and reusable across operations
        self.engine = self._create_engine()

    def _create_engine(self):

        #Create SQLAlchemy engine for database connections
        user = self.config["user"]
        password = self.config["password"]
        host = self.config["host"]
        port = self.config["port"]
        database = self.config["database"]

        # Build connection string from components
        return create_engine(
            f"mariadb+mariadbconnector://{user}:{password}@{host}:{port}/{database}"
        )

    @contextmanager
    def get_connection(self):
        #Context manager for database connections

        try:
            # Try preferred MariaDB connector first
            import mariadb
            conn = mariadb.connect(**self.config)
        except ImportError:
            # Fallback to PyMySQL if MariaDB connector not installed
            import pymysql
            conn = pymysql.connect(**self.config, charset="utf8mb4")

        try:
            # Yield connection to caller
            yield conn
        finally:
            # Close connection when done, even if exception occurs
            conn.close()

    def query(self, sql: str, params: Optional[Dict] = None) -> pd.DataFrame:
        # Execute SQL query and return results as pandas DataFrame
        # pd.read_sql handles:
        # - Query execution
        # - Result fetching
        # - Type conversion to appropriate pandas dtypes
        # - Memory-efficient chunked reading for large results
        return pd.read_sql(sql, self.engine, params=params)

    def setup_temperature_table(self) -> None:
        #Create database schema for temperature data
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Create database if it doesn't exist
            cursor.execute("CREATE DATABASE IF NOT EXISTS group06")

            # Switch to group database
            cursor.execute("USE group06")

            # Drop existing table to ensure clean slate
            cursor.execute("DROP TABLE IF EXISTS temperature")

            # Create table with optimized structure
            cursor.execute("""
                CREATE TABLE temperature (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    area VARCHAR(200),
                    month VARCHAR(20),
                    element VARCHAR(100),
                    year SMALLINT,
                    value DOUBLE,
                    INDEX idx_area_year (area, year),
                    INDEX idx_element_year (element, year),
                    INDEX idx_month (month)
                ) ENGINE=InnoDB
            """)

            # Commit transaction to persist changes
            conn.commit()

    def load_and_transform_data(self, source_table: str) -> pd.DataFrame:
        #Load wide-format data from database and transform to long format
        # SECURITY CHECK: Validate table name against whitelist
        if source_table not in ALLOWED_SOURCE_TABLES:
            raise ValueError(
                f"Invalid table name: {source_table}. "
                f"Allowed tables: {', '.join(ALLOWED_SOURCE_TABLES)}"
            )

        # Load table into memory
        query = f"SELECT * FROM {source_table}"
        df = self.query(query)

        # Identify year columns by pattern matching
        # Year columns follow format: Y1961, Y1962, ad infinitum
        year_cols = [c for c in df.columns
                     if c.strip().upper().startswith('Y')
                     and c.strip()[1:].isdigit()]

        # Columns that identify each unique time series
        id_cols = ['Area', 'Months', 'Element']

        df_long = df.melt(
            id_vars=id_cols,  # Keep these columns as-is
            value_vars=year_cols,  # These columns become rows
            var_name='year_col',  # Name for new column holding year labels
            value_name='value'  # Name for new column holding temperatures
        )

        # Extract year number from column name (Y1961 → 1961)
        df_long['year'] = df_long['year_col'].str[1:].astype(int)

        # Convert value column to numeric, replacing errors with NaN
        df_long['value'] = pd.to_numeric(df_long['value'], errors='coerce')

        # Remove rows with missing values
        df_long = df_long.dropna(subset=['value'])

        # Return cleaned dataframe with standardized column names
        return df_long[['Area', 'Months', 'Element', 'year', 'value']].rename(
            columns={'Area': 'area', 'Months': 'month', 'Element': 'element'}
        )

    def import_data(self, df: pd.DataFrame) -> None:

        #Import DataFrame to temperature table using batch insertion
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Prepared statement with placeholders
            sql = "INSERT INTO temperature (area, month, element, year, value) VALUES (%s, %s, %s, %s, %s)"

            # Convert DataFrame to list of tuples
            rows = list(df.itertuples(index=False, name=None))

            # Insert in batches to manage memory and transaction size
            for i in range(0, len(rows), BATCH_SIZE):
                # Get batch: rows[i] to rows[i+BATCH_SIZE]
                cursor.executemany(sql, rows[i:i + BATCH_SIZE])

                # Commit after each batch to avoid way to long transactions
                conn.commit()

            print(f"Imported {len(rows)} rows successfully")


# ==================== PLOTTING UTILITIES ====================

#Reusable utilities for consistent plot styling.

@contextmanager
def dark_plot_style():
    #Context manager for creating dark-themed plots

    # Create figure and axis with standard size
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Style the figure background
    fig.patch.set_facecolor(DARK_THEME["bg"])

    # Style the plot area
    ax.set_facecolor(DARK_THEME["plot_bg"])

    # Style all axis elements to be visible on dark background
    ax.tick_params(colors=DARK_THEME["text"])  # Tick marks and labels
    ax.xaxis.label.set_color(DARK_THEME["text"])  # X-axis label
    ax.yaxis.label.set_color(DARK_THEME["text"])  # Y-axis label
    ax.title.set_color(DARK_THEME["text"])  # Plot title

    # Yield control to caller who will add plot content
    yield fig, ax


def save_plot(filename: str, dark_mode: bool = False) -> None:
    #Save current matplotlib figure with consistent settings

    # Set background color based on theme
    facecolor = DARK_THEME["bg"] if dark_mode else 'white'

    # Save with optimized settings
    plt.savefig(
        filename,
        dpi=DPI,  # Resolution in dots per inch
        facecolor=facecolor,  # Background color
        bbox_inches='tight'  # Crop to content
    )

    # User feedback
    print(f"Saved: {filename}")


def clean_month_data(df: pd.DataFrame, month_col: str = 'Months') -> pd.DataFrame:
    #Clean and validate month column in DataFrame

    # Make a copy to avoid modifying original DataFrame
    df = df.copy()

    # Chain string operations for cleaning:
    df[month_col] = df[month_col].str.strip().str.replace('^-+', '', regex=True)

    # Filter to only valid month names
    return df[df[month_col].isin(MONTHS)]


def get_color_map(n_colors: Optional[int] = None) -> LinearSegmentedColormap:
    #Create temperature color map from predefined color list
    if n_colors:
        # Create colormap with exact number of colors
        return LinearSegmentedColormap.from_list("temp_change", TEMP_COLORS, N=n_colors)

    # Create continuous colormap with smooth gradients
    return LinearSegmentedColormap.from_list("temp_change", TEMP_COLORS)


def brighten_color(color: Tuple[float, float, float, float],
                   factor: float = 1.4) -> Tuple[float, float, float, float]:

    # Brighten an RGBA color by scaling RGB components
    r, g, b, a = color  # Unpack color components

    # Scale each color channel and cap at maximum
    return (
        min(r * factor, 1),  # Red
        min(g * factor, 1),  # Green
        min(b * factor, 1),  # Blue
        a  # Alpha (unchanged)
    )


# ==================== ANALYSIS QUERIES ====================
class TemperatureAnalyzer:
    #Handles all temperature data analysis queries
    def __init__(self, db_manager: DatabaseManager):
        #Initialize analyzer with database manager
        self.db = db_manager

    def winter_vs_summer(self) -> pd.DataFrame:
        #Compare winter and summer temperature changes over time
        df = self.db.query("""
            SELECT year,
                AVG(CASE WHEN month IN ('December', 'January', 'February') 
                    THEN value END) AS winter_avg,
                AVG(CASE WHEN month IN ('June', 'July', 'August') 
                    THEN value END) AS summer_avg
            FROM temperature
            WHERE element = 'Temperature change'
            GROUP BY year ORDER BY year
        """)

        # Data validation: Ensure query returned results
        if df.empty:
            raise ValueError("No data found for winter vs summer analysis")

        # Calculate seasonal difference for further analysis
        df['diff'] = df['summer_avg'] - df['winter_avg']
        return df

    def country_acceleration(self, top_n: int = 12) -> pd.DataFrame:
        # Find countries with highest temperature acceleration
        """
        SQL Math Breakdown:
        This implements the least squares regression formula:
        slope = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)

        Where:
        - n = COUNT(*) = number of data points
        - x = year
        - y = temp (temperature)
        - Σxy = SUM(year * temp)
        - Σx = SUM(year)
        - Σy = SUM(temp)
        - Σx² = SUM(year * year)
        """

        df = self.db.query(f"""
            WITH yearly AS (
                SELECT area, year, AVG(value) AS temp
                FROM temperature
                WHERE element = 'Temperature change'
                GROUP BY area, year
            )
            SELECT area,
                (COUNT(*) * SUM(year * temp) - SUM(year) * SUM(temp)) / 
                (COUNT(*) * SUM(year * year) - SUM(year) * SUM(year)) AS slope
            FROM yearly
            GROUP BY area
            HAVING COUNT(*) > 10
            ORDER BY slope DESC
            LIMIT {top_n}
        """)

        if df.empty:
            raise ValueError("No data found for country acceleration analysis")

        return df

    def global_temperature(self) -> pd.DataFrame:
        #Get global average temperature change time series
        df = self.db.query("""
            SELECT year, AVG(value) AS temp
            FROM temperature
            WHERE element = 'Temperature change'
            GROUP BY year ORDER BY year
        """)

        if df.empty:
            raise ValueError("No data found for global temperature analysis")

        return df

    def monthly_warming(self) -> pd.DataFrame:

        #Calculate warming trend for each month separately
        df = self.db.query("""
            WITH monthly AS (
                SELECT month, year, AVG(value) AS temp
                FROM temperature
                WHERE element = 'Temperature change'
                  AND month IN ('January', 'February', 'March', 'April', 'May', 'June',
                               'July', 'August', 'September', 'October', 'November', 'December')
                GROUP BY month, year
            )
            SELECT month,
                (COUNT(*) * SUM(year * temp) - SUM(year) * SUM(temp)) / 
                (COUNT(*) * SUM(year * year) - SUM(year) * SUM(year)) AS slope
            FROM monthly
            GROUP BY month
            HAVING COUNT(*) > 10
        """)

        if df.empty:
            raise ValueError("No data found for monthly warming analysis")

        # Convert month to categorical for proper sorting
        df['month'] = pd.Categorical(df['month'], categories=MONTHS, ordered=True)
        return df.sort_values('month')

    def get_long_format_data(self) -> pd.DataFrame:
        #Get all temperature data in long format with month numbers

        # Load temperature table
        df = self.db.query("SELECT area, month, element, year, value FROM temperature")

        if df.empty:
            raise ValueError("No data found in temperature table")

        # Standardize column names
        df.columns = ['Area', 'Months', 'Element', 'Year', 'Value']

        # Add numerical month column for sorting and calculations
        # map() looks up each month name in MONTH_MAP dictionary
        df['Month_Num'] = df['Months'].map(MONTH_MAP)

        # Filter out any rows where month mapping failed
        # notna() checks for non-NaN values
        return df[df['Month_Num'].notna()]

    def get_area_data(self, area: str, element: str = 'Temperature change',
                      year_start: Optional[int] = None,
                      year_end: Optional[int] = None) -> pd.DataFrame:

        #Get temperature data for specific area with optional filtering

        # Build WHERE clause conditions dynamically
        conditions = [
            "area = %(area)s",  # Always filter by area
            "element = %(element)s"  # Always filter by element type
        ]

        # Start with required parameters
        params = {"area": area, "element": element}

        # Add optional year filters if provided
        if year_start:
            conditions.append("year >= %(year_start)s")
            params["year_start"] = year_start
        if year_end:
            conditions.append("year <= %(year_end)s")
            params["year_end"] = year_end

        # Join conditions with AND
        where_clause = " AND ".join(conditions)

        # Build final query with dynamic WHERE clause
        query = f"""
            SELECT month as Months, element as Element, year as Year, value as Temperature
            FROM temperature
            WHERE {where_clause}
        """

        # Execute parameterized query
        df = self.db.query(query, params)

        # Informative warning (not error) if no data found
        # Allows program to continue gracefully
        if df.empty:
            print(f"Warning: No data found for area '{area}'")

        return df


# ==================== PLOTTING FUNCTIONS ====================
class PlotGenerator:

    #Generates all 15 temperature analysis visualizations
    def __init__(self, analyzer: TemperatureAnalyzer):
        #Initialize plot generator with data analyzer
        self.analyzer = analyzer

    def plot_line(self, df: pd.DataFrame, x: str, y, title: str,
                  xlabel: str, ylabel: str, filename: str) -> None:


        #Generic line plot with support for single or multiple series
        plt.figure(figsize=FIGURE_SIZE)

        # Handle multiple series
        if isinstance(y, list):
            for col, label in y:
                # Plot each series with legend entry
                plt.plot(df[x], df[col], label=label, linewidth=2)
            plt.legend()  # Show legend when multiple series
        else:
            # Single series
            plt.plot(df[x], df[y], linewidth=2)

        # Apply standard styling
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(alpha=0.3)  # Subtle grid for readability
        save_plot(filename)
        plt.show()  # Display interactively
        plt.close()  # Free memory

    def plot_bar(self, df: pd.DataFrame, x: str, y: str, title: str,
                 xlabel: str, ylabel: str, filename: str,
                 horizontal: bool = False) -> None:

        #Generic bar plot with horizontal/vertical options
        plt.figure(figsize=FIGURE_SIZE)

        if horizontal:
            # Horizontal bars (y-axis shows categories)
            plt.barh(range(len(df)), df[y], color='coral')
            plt.yticks(range(len(df)), df[x], fontsize=9)
        else:
            # Vertical bars (x-axis shows categories)
            plt.bar(range(len(df)), df[y], color='orangered')
            # Rotate labels 45°, align right
            plt.xticks(range(len(df)), df[x], rotation=45, ha='right', fontsize=9)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title, wrap=True)  # Wrap long titles
        plt.grid(alpha=0.3)
        plt.tight_layout()  # Adjust spacing to prevent label cutoff
        save_plot(filename)
        plt.show()
        plt.close()

    def generate_all_graphs(self) -> None:
        #Master function to generate all 15 graphs

        try:
            self._generate_program1_graphs()
            self._generate_program2_graphs()
            self._generate_program3_graphs()

            # Success message
            print("\n" + "=" * 60)
            print("ALL 15 GRAPHS COMPLETED SUCCESSFULLY!")
            print("=" * 60)
        except Exception as e:
            print(f"Error generating graphs: {e}")
            raise  # Re-raise to show full error details

    def _generate_program1_graphs(self) -> None:
        #Generate graphs 1-5: Fundamental climate analysis

        print("\n=== Generating Program 1 Graphs (1-5) ===")

        # Graph 1: Compare winter and summer warming rates
        print("Graph 1: Winter vs Summer...")
        df1 = self.analyzer.winter_vs_summer()
        self.plot_line(
            df1, 'year',
            [('winter_avg', 'Winter'), ('summer_avg', 'Summer')],  # Two lines
            'Winter vs Summer Temperature Change', 'Year',
            'Temperature Change (°C)', 'graph_01_winter_vs_summer.png'
        )

        # Graph 2: Countries with fastest temperature rise
        print("Graph 2: Country Acceleration...")
        df2 = self.analyzer.country_acceleration()
        self.plot_bar(
            df2.head(12),  # Top 12 countries
            'area', 'slope',
            'Top Countries by Temperature Acceleration',
            'Temperature Acceleration (°C/year)', 'Country',
            'graph_02_country_acceleration.png',
            horizontal=True  # Horizontal for readability
        )

        # Graph 3: Global temperature over time
        print("Graph 3: Thermal Inertia...")
        df3 = self.analyzer.global_temperature()
        self.plot_line(
            df3, 'year', 'temp', 'Global Temperature Change',
            'Year', 'Temperature Change (°C)', 'graph_03_thermal_inertia.png'
        )

        # Graph 4: Which months warm fastest
        print("Graph 4: Monthly Warming...")
        df4 = self.analyzer.monthly_warming()
        self.plot_bar(
            df4, 'month', 'slope', 'Monthly Warming Trends',
            'Month', 'Warming Rate (°C/year)', 'graph_04_monthly_warming.png'
        )

        # Graph 5: When will we hit critical thresholds?
        print("Graph 5: Temperature Projections...")
        self._plot_projection(df3)

    def _plot_projection(self, df: pd.DataFrame) -> None:
        #Plot when temperature will cross critical thresholds

        # Extract numpy arrays for mathematical operations
        x, y = df['year'].values, df['temp'].values

        # Fit linear regression
        slope, intercept = np.polyfit(x, y, 1)

        # Create future projection range
        years = np.arange(x.min(), x.max() + 100)

        plt.figure(figsize=FIGURE_SIZE)

        # Plot observed data as scatter points
        plt.plot(x, y, 'o', label='Observed', markersize=4)

        # Plot trend line
        plt.plot(years, slope * years + intercept, '--',
                 label=f'Trend ({slope:.5f} °C/year)', linewidth=2)

        # Add threshold lines and crossing points
        for thresh in TEMP_THRESHOLDS:
            # Calculate year when trend crosses threshold
            year_cross = (thresh - intercept) / slope

            # Draw horizontal threshold line
            plt.axhline(thresh, linestyle=':', label=f'{thresh}°C Threshold')

            # Mark crossing point with X
            plt.plot(year_cross, thresh, 'X', markersize=12)

            # Add text annotation
            plt.text(year_cross, thresh, f' Crossed in {int(year_cross)}')

        plt.xlabel('Year')
        plt.ylabel('Temperature Change (°C)')
        plt.title('Temperature Threshold Projections')
        plt.legend()
        plt.grid(alpha=0.3)
        save_plot('graph_05_threshold_projection.png')
        plt.show()
        plt.close()

    def _generate_program2_graphs(self) -> None:
        # Generate graphs 6-10: Advanced visualizations with dark theme

        print("\n=== Generating Program 2 Graphs (6-10) ===")

        # Load all data once for efficiency (used by multiple graphs)
        df_long = self.analyzer.get_long_format_data()

        print("Graph 6: Global Temperature Over Years (Animated)...")
        self._plot_global_temp_animated(df_long)

        print("Graph 7: Top 30 Areas in 2019...")
        self._plot_top_areas(df_long, interactive=True)

        print("Graph 8: China Temperature Changes...")
        self._plot_area_temp(df_long)

        print("Graph 9: Temperature Variability...")
        self._plot_temp_variability(df_long)

        print("Graph 10: Regional Comparison...")
        self._plot_regional_comparison(df_long)

    def _plot_global_temp_animated(self, df_long: pd.DataFrame) -> None:
        # Graph 6: Animated visualization of global temperature by month

        # Filter to World data only
        world_long = df_long[
            (df_long["Area"] == "World") &
            (df_long["Element"] == "Temperature change")
        ].copy()

        if world_long.empty:
            print("Warning: No World data found for animation")
            return

        # Pivot to wide format for animation
        # Each row = one year, columns = months 1–12
        pivot = world_long.pivot(
            index="Year", columns="Month_Num", values="Value"
        ).sort_index()

        # Prepare color scheme
        years = pivot.index.values
        color_map = get_color_map(len(years))
        # Assign one color per year
        year_colors = [color_map(i / len(years)) for i in range(len(years))]

        # Create figure with dark theme
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor(DARK_THEME["bg"])
        ax.set_facecolor(DARK_THEME["plot_bg"])

        # Add padding to prevent text cutoff
        fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)

        # 🔹 Add colorbar legend to show year mapping
        norm = Normalize(vmin=years.min(), vmax=years.max())
        sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
        cbar = plt.colorbar(sm, ax=ax, orientation="vertical", pad=0.02)
        cbar.set_label("Year", fontsize=10, color=DARK_THEME["text"])
        cbar.ax.tick_params(colors=DARK_THEME["text"], labelcolor=DARK_THEME["text"])

        def update(frame):
            # Update function called for each animation frame
            ax.clear()  # Clear previous frame

            # Restore dark theme (cleared by ax.clear())
            ax.set_facecolor(DARK_THEME["plot_bg"])

            # Set axis properties
            ax.set_xlim(0, 11)  # Months 0–11
            ax.set_ylim(-2, 3)  # Temperature range
            ax.set_xticks(range(12))
            ax.set_xticklabels(MONTH_ABBREV, color=DARK_THEME["text"], fontsize=10)
            ax.set_xlabel("Month", color=DARK_THEME["text"], fontsize=12)
            ax.set_ylabel("Temperature Change (°C)", color=DARK_THEME["text"], fontsize=12)
            ax.tick_params(colors=DARK_THEME["text"], labelsize=10)

            current_year = years[frame]

            # Draw all previous years as faint background traces
            for i in range(frame):
                ax.plot(range(12), pivot.iloc[i],
                        color=year_colors[i], alpha=0.2, linewidth=1.5)

            # Create glow effect for current year
            bright_color = brighten_color(year_colors[frame])

            # Soft halo effect layers
            for width, alpha in [(11, 0.05), (9, 0.08), (7, 0.1)]:
                ax.plot(range(12), pivot.iloc[frame],
                        color=bright_color, linewidth=width, alpha=alpha)

            # Main line on top (brightest)
            ax.plot(range(12), pivot.iloc[frame],
                    color=bright_color, linewidth=3.5, alpha=1.0)

            # Update title with current year
            ax.set_title(f"Global Temperature Change – {current_year}",
                         fontsize=16, weight="bold", color=DARK_THEME["text"], pad=20)

        # Create animation
        ani = FuncAnimation(
            fig, update,
            frames=len(years),  # One frame per year
            interval=400,       # 400 ms between frames (2.5 fps)
            repeat=True         # Loop animation
        )

        # Save as animated GIF
        try:
            ani.save(
                'graph_06_global_temp_animated.gif',
                writer='pillow',  # GIF writer
                fps=2,  # 2 frames per second
                savefig_kwargs={'facecolor': DARK_THEME["bg"], 'pad_inches': 0.3}
            )
            print("Saved: graph_06_global_temp_animated.gif")
        except Exception as e:
            print(f"Could not save animation: {e}")

        plt.show()
        plt.close()


    def _plot_top_areas(self, df_long: pd.DataFrame, year: int = None, top_n: int = TOP_AREAS_COUNT,
                        interactive: bool = False) -> None:
        # Graph 7: Top N areas by temperature change in specific year

        # If interactive mode, ask user for year
        # (..-. --.- .... / -- .-. -- / .... -..- -.. / ...- .--- - -. / ...- -. / .--- -- -- / -.-. --.- .-. -... / .--- -.-. / -.-. --.- -. / ..- .--- -... -.-. / ...- .-. .-- -.. -.-. -. / .-- .--- --.- .-. -..- ...- .... --..-- / -- -..- / .... -..- -.. / --.- .--- -.-. -. / ...- -. ..--.. / .--. -..- -- -- .--- ...- .-- / -. .-- -.-. .-. -.-. ..- -. -- / -.- .- .--- -.-. .-.-.-)
        #(C & M)

        if interactive or year is None:
            while True:
                try:
                    year_input = input("Enter a year between 1961 and 2019: ")
                    year = int(year_input)
                    if 1961 <= year <= 2019:
                        break
                    else:
                        print("Year must be between 1961 and 2019. Try again.")
                except ValueError:
                    print("Invalid input. Enter a number.")

        # Filter to temperature change data only
        regions_long = df_long[df_long["Element"] == "Temperature change"].copy()
        df_year = regions_long[regions_long["Year"] == year]

        if df_year.empty:
            print(f"Warning: No data found for year {year}")
            return

        # Calculate average temperature per area, get top N
        area_avg = df_year.groupby("Area")["Value"].mean().sort_values(
            ascending=False
        ).head(top_n)

        # Create color map based on temperature values
        color_map = get_color_map()
        norm = Normalize(vmin=area_avg.min(), vmax=area_avg.max())
        bar_colors = [color_map(norm(val)) for val in area_avg.values]

        # Create dark-themed figure
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor(DARK_THEME["bg"])
        ax.set_facecolor(DARK_THEME["plot_bg"])
        ax.tick_params(colors=DARK_THEME["text"])
        ax.xaxis.label.set_color(DARK_THEME["text"])
        ax.yaxis.label.set_color(DARK_THEME["text"])
        ax.title.set_color(DARK_THEME["text"])

        # Plot bars with temperature-based colors
        ax.bar(area_avg.index, area_avg.values, color=bar_colors)
        ax.set_xticks(range(len(area_avg.index)))
        ax.set_xticklabels(area_avg.index, rotation=45, ha="right")
        ax.set_ylabel("Average Temperature Change (°C)")
        ax.set_title(f"Top {top_n} Areas with Highest Avg Temperature Change in {year}")

        plt.tight_layout()
        save_plot(f'graph_07_top30_areas_{year}.png', dark_mode=True)
        plt.show()
        plt.close()

    def _plot_area_temp(self, df_long: pd.DataFrame) -> None:
        #Graph 8: Multi-year monthly temperature for specific area

        regions_long = df_long[df_long["Element"] == "Temperature change"].copy()
        all_areas = regions_long["Area"].unique().tolist()
        area = random.choice(all_areas)

        df_area = regions_long[regions_long["Area"] == area]

        if df_area.empty:
            print(f"Warning: No data found for area '{area}'")
            return

        # Pivot: rows=years, columns=months
        pivot_area = df_area.pivot(
            index="Year", columns="Month_Num", values="Value"
        ).sort_index()

        years_area = pivot_area.index.values
        color_map_area = get_color_map(len(years_area))
        # One color per year (temporal progression)
        year_colors = [color_map_area(i / len(years_area))
                       for i in range(len(years_area))]

        # Create dark-themed figure
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor(DARK_THEME["bg"])
        ax.set_facecolor(DARK_THEME["plot_bg"])
        ax.tick_params(colors=DARK_THEME["text"])
        ax.xaxis.label.set_color(DARK_THEME["text"])
        ax.yaxis.label.set_color(DARK_THEME["text"])
        ax.title.set_color(DARK_THEME["text"])

        # Set up monthly x-axis
        ax.set_xlim(0, 11)
        ax.set_xticks(range(12))
        ax.set_xticklabels(MONTH_ABBREV)
        ax.set_ylabel("Temperature Change (°C)")
        ax.set_xlabel("Month")
        ax.set_title(f"Monthly Temperature Changes in {area} Over the Years",
                     fontsize=14, weight="bold")

        # Plot one line per year
        for i in range(len(years_area)):
            ax.plot(range(12), pivot_area.iloc[i],
                    color=year_colors[i], linewidth=2, alpha=0.7)

        # Add colorbar to show year mapping
        norm_area = Normalize(vmin=years_area.min(), vmax=years_area.max())
        sm_area = plt.cm.ScalarMappable(cmap=color_map_area, norm=norm_area)
        cbar_area = plt.colorbar(sm_area, ax=ax, orientation="vertical", pad=0.02)
        cbar_area.set_label("Year", fontsize=10, color=DARK_THEME["text"])
        cbar_area.ax.tick_params(colors=DARK_THEME["text"], labelcolor=DARK_THEME["text"])

        plt.tight_layout()
        save_plot('graph_08_china_temp_changes.png', dark_mode=True)
        plt.show()
        plt.close()

    def _plot_temp_variability(self, df_long: pd.DataFrame) -> None:
        #Graph 9: Temperature variability over time

        # Filter to standard deviation data for World
        std_data = df_long[
            (df_long["Element"] == "Standard Deviation") &
            (df_long["Area"] == "World")
            ].copy()

        if std_data.empty:
            print("Warning: No Standard Deviation data found for World")
            return

        # Create dark-themed figure
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor(DARK_THEME["bg"])
        ax.set_facecolor(DARK_THEME["plot_bg"])
        ax.tick_params(colors=DARK_THEME["text"])
        ax.xaxis.label.set_color(DARK_THEME["text"])
        ax.yaxis.label.set_color(DARK_THEME["text"])
        ax.title.set_color(DARK_THEME["text"])

        ax.set_xlabel("Month")
        ax.set_ylabel("Temperature Variability (°C)")
        ax.set_title("Monthly Temperature Variability Over the Years",
                     fontsize=14, weight="bold")
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(MONTH_ABBREV)

        # Plot one line per year
        for year in sorted(std_data["Year"].unique()):
            yearly_data = std_data[std_data["Year"] == year]
            ax.plot(yearly_data["Month_Num"], yearly_data["Value"],
                    color="lightsteelblue", linewidth=2.5)

        plt.tight_layout()
        save_plot('graph_09_temp_variability.png', dark_mode=True)
        plt.show()
        plt.close()

    def _plot_regional_comparison(self, df_long: pd.DataFrame) -> None:
        #Graph 10: Compare temperature trends across major regions


        areas_to_compare = ["World", "Europe", "Asia", "Americas"]
        area_colors = {
            "World": "#a11216",  # Red (hottest in color scheme)
            "Europe": "#f89c56",  # Orange
            "Asia": "#c5e7ad",  # Light green
            "Americas": "#64aaaf"  # Teal
        }

        # Filter to temperature change for selected regions
        regions_long = df_long[df_long["Element"] == "Temperature change"].copy()
        temp_compare = regions_long[regions_long["Area"].isin(areas_to_compare)]

        if temp_compare.empty:
            print("Warning: No data found for regional comparison")
            return

        # Calculate yearly averages for each region
        area_yearly = temp_compare.groupby(["Area", "Year"])["Value"].mean().reset_index()

        # Create dark-themed figure
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor(DARK_THEME["bg"])
        ax.set_facecolor(DARK_THEME["plot_bg"])
        ax.tick_params(colors=DARK_THEME["text"])
        ax.xaxis.label.set_color(DARK_THEME["text"])
        ax.yaxis.label.set_color(DARK_THEME["text"])
        ax.title.set_color(DARK_THEME["text"])

        ax.set_xlabel("Year")
        ax.set_ylabel("Average Temperature Change (°C)")
        ax.set_title("Comparison of Temperature Change Across Regions (1961–2019)",
                     fontsize=14, weight="bold")

        # Plot each region with assigned color
        for area in areas_to_compare:
            data = area_yearly[area_yearly["Area"] == area]
            ax.plot(data["Year"], data["Value"], label=area,
                    color=area_colors.get(area, "white"), linewidth=2)

        # Style legend for dark theme
        legend = ax.legend(facecolor=DARK_THEME["plot_bg"],
                           edgecolor=DARK_THEME["text"])
        for text in legend.get_texts():
            text.set_color(DARK_THEME["text"])

        plt.tight_layout()
        save_plot('graph_10_regional_comparison.png', dark_mode=True)
        plt.show()
        plt.close()

    def _generate_program3_graphs(self) -> None:
        #Generate graphs 11-15: Region-specific detailed analysis

        print("\n=== Generating Program 3 Graphs (11-15) ===")

        try:
            print("Graph 11: Afghanistan Temperature (1980-2000)...")
            self._plot_afghanistan_temp()

            print("Graph 12: Monthly Variation...")
            self._plot_month_variation()

            print("Graph 13: Seasonal Trends...")
            self._plot_seasonal_trend()

            print("Graph 14: Decade Trends...")
            self._plot_decade_trend()

            print("Graph 15: Month Comparison All Areas...")
            self._plot_month_comparison_all()

        except Exception as e:
            print(f"Error in Program 3 graphs: {e}")
            raise  # Re-raise for full traceback

    def _plot_afghanistan_temp(self) -> None:
        #Graph 11: Afghanistan temperature by month (1980-2000)

        # Query specific area and date range
        df = self.analyzer.get_area_data('Afghanistan', year_start=1980, year_end=2000)

        if df.empty:
            print("Warning: No data found for Afghanistan (1980-2000)")
            return

        # Clean month data and ensure year is integer
        df = clean_month_data(df)
        df['Year'] = df['Year'].astype(int)

        plt.figure(figsize=FIGURE_SIZE)

        # Get unique months and assign colors
        months = df['Months'].unique()
        colors_palette = cm.viridis(np.linspace(0, 1, len(months)))

        # Plot one line per month
        for i, month in enumerate(months):
            month_df = df[df['Months'] == month]
            plt.plot(month_df['Year'], month_df['Temperature'],
                     label=month, color=colors_palette[i], linewidth=2)

        plt.title("Average Temperature Change per Month in Afghanistan (1980–2000)")
        plt.xlabel("Year")
        plt.ylabel("Temperature Change (°C)")
        # Place legend outside plot area to avoid obscuring data
        plt.legend(title="Month", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        save_plot('graph_11_afghanistan_temp.png')
        plt.show()
        plt.close()

    def _plot_month_variation(self) -> None:
        #Graph 12: Which months have most temperature variation globally

        # Query all temperature change data in date range
        query = """
            SELECT month as Months, value as Temperature
            FROM temperature
            WHERE element='Temperature change'
            AND year BETWEEN 1961 AND 2019
        """
        df = self.analyzer.db.query(query)

        if df.empty:
            print("Warning: No data found for monthly variation")
            return

        # Clean month names
        df = clean_month_data(df)

        # Calculate standard deviation per month
        df_std = df.groupby('Months')['Temperature'].std().reset_index()
        df_std.columns = ['Months', 'std']

        # Convert to categorical for proper ordering
        df_std['Months'] = pd.Categorical(df_std['Months'],
                                          categories=MONTHS, ordered=True)
        # Sort by std dev (highest variation first)
        df_sorted = df_std.sort_values('std', ascending=False)

        plt.figure(figsize=FIGURE_SIZE)
        # Color gradient based on position
        colors_palette = plt.cm.viridis(np.linspace(0, 1, len(df_sorted)))
        plt.bar(df_sorted['Months'].astype(str), df_sorted['std'],
                color=colors_palette)
        plt.title("Months with Most Temperature Variation")
        plt.xlabel("Month")
        plt.ylabel("Standard Deviation of Temperature (°C)")
        plt.xticks(rotation=45)  # Angle labels for readability
        plt.tight_layout()
        save_plot('graph_12_month_variation.png')
        plt.show()
        plt.close()

    def _plot_seasonal_trend(self) -> None:
        #Graph 13: Seasonal temperature trends in Afghanistan

        # Get all Afghanistan data
        df = self.analyzer.get_area_data('Afghanistan', year_start=1961, year_end=2019)

        if df.empty:
            print("Warning: No data found for Afghanistan seasonal trends")
            return

        # Clean and prepare data
        df = clean_month_data(df)
        df['Year'] = df['Year'].astype(int)

        # Create month-to-season mapping
        month_to_season = {}
        for season, months in SEASONS.items():
            for month in months:
                month_to_season[month] = season

        # Add season column
        df['Season'] = df['Months'].map(month_to_season)

        # Pivot: rows=years, columns=seasons, values=avg temp
        df_pivot = df.pivot_table(index='Year', columns='Season',
                                  values='Temperature', aggfunc='mean')

        # Create stacked bar chart
        df_pivot.plot.bar(stacked=True, figsize=FIGURE_SIZE, colormap=cm.viridis,
                          title="Average Seasonal Temperature per Year in Afghanistan")
        plt.xlabel("Year")
        plt.ylabel("Temperature Change (°C)")
        plt.tight_layout()
        save_plot('graph_13_seasonal_trend.png')
        plt.show()
        plt.close()

    def _plot_decade_trend(self) -> None:
        #Graph 14: Temperature change by decade in Afghanistan

        # Get ALL Afghanistan data
        df = self.analyzer.get_area_data('Afghanistan')

        if df.empty:
            print("Warning: No data found for Afghanistan decade trends")
            return

        # Clean and filter to decade years only
        df = clean_month_data(df)
        df['Year'] = df['Year'].astype(int)
        df = df[df['Year'].isin(DECADE_YEARS)]

        # Calculate mean temperature per decade year
        df_decade = df.groupby('Year')['Temperature'].mean()

        plt.figure(figsize=(8, 5))  # Smaller size for fewer bars

        # Create color gradient
        colors_palette = plt.cm.viridis(np.linspace(0, 1, len(df_decade)))

        # Create bars with colors
        bars = plt.bar(df_decade.index.astype(str), df_decade.values,
                       color=colors_palette, edgecolor='black')

        # Apply different hatch pattern to each bar
        patterns = ['/', '\\', '|', '-', '+', 'x', 'o']
        for i, bar in enumerate(bars):
            bar.set_hatch(patterns[i % len(patterns)])

        plt.title("Average Temperature Change per Decade in Afghanistan")
        plt.xlabel("Year")
        plt.ylabel("Temperature Change (°C)")
        plt.tight_layout()
        save_plot('graph_14_decade_trend.png')
        plt.show()
        plt.close()

    def _plot_month_comparison_all(self) -> None:
        #Graph 15: Compare months across decades globally

        decades = [1961, 1970, 1980, 1990, 2000, 2010, 2019]

        # Query data for specific decades
        query = f"""
            SELECT month as Months, year as Year, value as Temperature
            FROM temperature
            WHERE element='Temperature change'
            AND year IN ({','.join(map(str, decades))})
        """
        df = self.analyzer.db.query(query)
        df = clean_month_data(df)
        df['Year'] = df['Year'].astype(str)  # Convert to string for categorical

        plt.figure(figsize=FIGURE_SIZE)

        # Create color palette
        palette = sns.color_palette("viridis", n_colors=len(df['Year'].unique()))

        # Seaborn automatically groups bars by 'hue' parameter
        # x='Months': Creates group for each month
        # hue='Year': Creates bar for each year within group
        sns.barplot(x='Months', y='Temperature', hue='Year', data=df, palette=palette)

        plt.title("Temperature Comparison Across Months")
        plt.xlabel("Month")
        plt.ylabel("Temperature Change (°C)")
        # Place legend outside to avoid obscuring data
        plt.legend(title="Year", bbox_to_anchor=(1, 1), loc='upper left')
        save_plot('graph_15_month_comparison_all.png')
        plt.show()
        plt.close()


# ==================== MAIN ====================
def main(source_table: str = 'temperature_data') -> None:
    """
    1. Initialize database connection
    2. Create/reset temperature table schema
    3. Load and transform data from source
    4. Import data in batches
    5. Generate all 15 visualizations
    """

    print("=" * 60)
    print("COMBINED TEMPERATURE ANALYSIS - 15 GRAPHS")
    print("=" * 60)

    try:
        # STEP 1: Initialize database connection
        db_manager = DatabaseManager(DB_CONFIG)

        # STEP 2: Setup database schema
        print("\nSetting up temperature table...")
        db_manager.setup_temperature_table()

        # STEP 3: Load and transform data
        print("Loading data from database table...")
        df = db_manager.load_and_transform_data(source_table)

        # STEP 4: Import transformed data
        print("Importing to temperature table...")
        db_manager.import_data(df)

        # STEP 5: Generate all visualizations
        print("\nStarting graph generation...")
        analyzer = TemperatureAnalyzer(db_manager)
        plot_generator = PlotGenerator(analyzer)
        plot_generator.generate_all_graphs()

    except ValueError as ve:
        # Handle validation errors
        print(f"\nValidation Error: {ve}")
        print("Please check your input parameters and try again.")
    except Exception as e:
        # Handle unexpected errors
        print(f"\nUnexpected Error: {e}")
        print("Please check your database connection and data.")
        raise  # Re-raise to show full traceback for debugging


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combined Temperature Analysis - 15 Graphs"
    )

    # Define command-line argument
    parser.add_argument(
        "--source",
        default="temperature_data",  # Default if not specified
        help=f"Source table name (allowed: {', '.join(ALLOWED_SOURCE_TABLES)})"
    )

    # Parse arguments from command line
    args = parser.parse_args()

    # Execute main function with provided source table
    main(args.source)