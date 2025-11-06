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

# ==================== CONSTANTS ====================
# Database configuration - should be loaded from environment variables in production
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "coen2220"),
    "password": os.getenv("DB_PASSWORD", "coen2220"),
    "database": os.getenv("DB_NAME", "group06"),
}

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

MONTH_ABBREV = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

MONTH_MAP = {month: i + 1 for i, month in enumerate(MONTHS)}

TEMP_COLORS = ["#1f628c", "#64aaaf", "#a2deaf", "#c5e7ad", "#fff4c6",
               "#fdc87f", "#f89c56", "#e65a3a", "#a11216"]

DARK_THEME = {
    "bg": "#0b0c10",
    "plot_bg": "#11141a",
    "text": "white"
}

SEASONS = {
    "Winter": ["December", "January", "February"],
    "Spring": ["March", "April", "May"],
    "Summer": ["June", "July", "August"],
    "Autumn": ["September", "October", "November"]
}

FIGURE_SIZE = (12, 6)
DPI = 100
BATCH_SIZE = 5000
TOP_AREAS_COUNT = 30
TEMP_THRESHOLDS = [1.5, 2.0]
DECADE_YEARS = [1961, 1970, 1980, 1990, 2000, 2010, 2019]

# Whitelist of allowed source tables
ALLOWED_SOURCE_TABLES = ["temperature_data", "climate_data", "temp_raw"]


# ==================== DATABASE ====================
class DatabaseManager:
    """Manages database connections and operations"""

    def __init__(self, config: dict):
        self.config = config
        self.engine = self._create_engine()

    def _create_engine(self):
        """Create SQLAlchemy engine"""
        user = self.config["user"]
        password = self.config["password"]
        host = self.config["host"]
        port = self.config["port"]
        database = self.config["database"]
        return create_engine(
            f"mariadb+mariadbconnector://{user}:{password}@{host}:{port}/{database}"
        )

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        try:
            import mariadb
            conn = mariadb.connect(**self.config)
        except ImportError:
            import pymysql
            conn = pymysql.connect(**self.config, charset="utf8mb4")

        try:
            yield conn
        finally:
            conn.close()

    def query(self, sql: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """Execute query and return DataFrame"""
        return pd.read_sql(sql, self.engine, params=params)

    def setup_temperature_table(self) -> None:
        """Create database and temperature table"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE DATABASE IF NOT EXISTS group06")
            cursor.execute("USE group06")
            cursor.execute("DROP TABLE IF EXISTS temperature")
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
            conn.commit()

    def load_and_transform_data(self, source_table: str) -> pd.DataFrame:
        """Load wide-format data and transform to long format"""
        # Validate source table against whitelist
        if source_table not in ALLOWED_SOURCE_TABLES:
            raise ValueError(
                f"Invalid table name: {source_table}. "
                f"Allowed tables: {', '.join(ALLOWED_SOURCE_TABLES)}"
            )

        query = f"SELECT * FROM {source_table}"
        df = self.query(query)

        year_cols = [c for c in df.columns
                     if c.strip().upper().startswith('Y')
                     and c.strip()[1:].isdigit()]

        id_cols = ['Area', 'Months', 'Element']
        df_long = df.melt(
            id_vars=id_cols,
            value_vars=year_cols,
            var_name='year_col',
            value_name='value'
        )

        df_long['year'] = df_long['year_col'].str[1:].astype(int)
        df_long['value'] = pd.to_numeric(df_long['value'], errors='coerce')
        df_long = df_long.dropna(subset=['value'])

        return df_long[['Area', 'Months', 'Element', 'year', 'value']].rename(
            columns={'Area': 'area', 'Months': 'month', 'Element': 'element'}
        )

    def import_data(self, df: pd.DataFrame) -> None:
        """Import data to temperature table in batches"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            sql = "INSERT INTO temperature (area, month, element, year, value) VALUES (%s, %s, %s, %s, %s)"

            rows = list(df.itertuples(index=False, name=None))
            for i in range(0, len(rows), BATCH_SIZE):
                cursor.executemany(sql, rows[i:i + BATCH_SIZE])
                conn.commit()
            print(f"Imported {len(rows)} rows successfully")


# ==================== PLOTTING UTILITIES ====================
@contextmanager
def dark_plot_style():
    """Context manager for dark-themed plots"""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    fig.patch.set_facecolor(DARK_THEME["bg"])
    ax.set_facecolor(DARK_THEME["plot_bg"])
    ax.tick_params(colors=DARK_THEME["text"])
    ax.xaxis.label.set_color(DARK_THEME["text"])
    ax.yaxis.label.set_color(DARK_THEME["text"])
    ax.title.set_color(DARK_THEME["text"])

    yield fig, ax


def save_plot(filename: str, dark_mode: bool = False) -> None:
    """Save plot with consistent settings"""
    facecolor = DARK_THEME["bg"] if dark_mode else 'white'
    plt.savefig(filename, dpi=DPI, facecolor=facecolor, bbox_inches='tight')
    print(f"Saved: {filename}")


def clean_month_data(df: pd.DataFrame, month_col: str = 'Months') -> pd.DataFrame:
    """Clean and filter month column"""
    df = df.copy()
    df[month_col] = df[month_col].str.strip().str.replace('^-+', '', regex=True)
    return df[df[month_col].isin(MONTHS)]


def get_color_map(n_colors: Optional[int] = None) -> LinearSegmentedColormap:
    """Get temperature color map"""
    if n_colors:
        return LinearSegmentedColormap.from_list("temp_change", TEMP_COLORS, N=n_colors)
    return LinearSegmentedColormap.from_list("temp_change", TEMP_COLORS)


def brighten_color(color: Tuple[float, float, float, float],
                   factor: float = 1.4) -> Tuple[float, float, float, float]:
    """Brighten a color by multiplying RGB values"""
    r, g, b, a = color
    return min(r * factor, 1), min(g * factor, 1), min(b * factor, 1), a


# ==================== ANALYSIS QUERIES ====================
class TemperatureAnalyzer:
    """Handles temperature data analysis queries"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def winter_vs_summer(self) -> pd.DataFrame:
        """Compare winter and summer temperature changes"""
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

        if df.empty:
            raise ValueError("No data found for winter vs summer analysis")

        df['diff'] = df['summer_avg'] - df['winter_avg']
        return df

    def country_acceleration(self, top_n: int = 12) -> pd.DataFrame:
        """Find countries with highest temperature acceleration"""
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
        """Get global temperature time series"""
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
        """Calculate warming trend by month"""
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

        df['month'] = pd.Categorical(df['month'], categories=MONTHS, ordered=True)
        return df.sort_values('month')

    def get_long_format_data(self) -> pd.DataFrame:
        """Get all data in long format with month numbers"""
        df = self.db.query("SELECT area, month, element, year, value FROM temperature")

        if df.empty:
            raise ValueError("No data found in temperature table")

        df.columns = ['Area', 'Months', 'Element', 'Year', 'Value']
        df['Month_Num'] = df['Months'].map(MONTH_MAP)
        return df[df['Month_Num'].notna()]

    def get_area_data(self, area: str, element: str = 'Temperature change',
                      year_start: Optional[int] = None,
                      year_end: Optional[int] = None) -> pd.DataFrame:
        """Get temperature data for specific area"""
        conditions = [
            "area = %(area)s",
            "element = %(element)s"
        ]

        params = {"area": area, "element": element}

        if year_start:
            conditions.append("year >= %(year_start)s")
            params["year_start"] = year_start
        if year_end:
            conditions.append("year <= %(year_end)s")
            params["year_end"] = year_end

        where_clause = " AND ".join(conditions)

        query = f"""
            SELECT month as Months, element as Element, year as Year, value as Temperature
            FROM temperature
            WHERE {where_clause}
        """
        df = self.db.query(query, params)

        if df.empty:
            print(f"Warning: No data found for area '{area}'")

        return df


# ==================== PLOTTING FUNCTIONS ====================
class PlotGenerator:
    """Generates all temperature analysis plots"""

    def __init__(self, analyzer: TemperatureAnalyzer):
        self.analyzer = analyzer

    def plot_line(self, df: pd.DataFrame, x: str, y, title: str,
                  xlabel: str, ylabel: str, filename: str) -> None:
        """Generic line plot"""
        plt.figure(figsize=FIGURE_SIZE)

        if isinstance(y, list):
            for col, label in y:
                plt.plot(df[x], df[col], label=label, linewidth=2)
            plt.legend()
        else:
            plt.plot(df[x], df[y], linewidth=2)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(alpha=0.3)
        save_plot(filename)
        plt.show()
        plt.close()

    def plot_bar(self, df: pd.DataFrame, x: str, y: str, title: str,
                 xlabel: str, ylabel: str, filename: str,
                 horizontal: bool = False) -> None:
        """Generic bar plot"""
        plt.figure(figsize=FIGURE_SIZE)

        if horizontal:
            plt.barh(range(len(df)), df[y], color='coral')
            plt.yticks(range(len(df)), df[x])
        else:
            plt.bar(range(len(df)), df[y], color='orangered')
            plt.xticks(range(len(df)), df[x], rotation=45, ha='right')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(alpha=0.3)
        save_plot(filename)
        plt.show()
        plt.close()

    def generate_all_graphs(self) -> None:
        """Generate all 15 graphs"""
        try:
            self._generate_program1_graphs()
            self._generate_program2_graphs()
            self._generate_program3_graphs()
            print("\n" + "=" * 60)
            print("ALL 15 GRAPHS COMPLETED SUCCESSFULLY!")
            print("=" * 60)
        except Exception as e:
            print(f"Error generating graphs: {e}")
            raise

    def _generate_program1_graphs(self) -> None:
        """Generate graphs 1-5"""
        print("\n=== Generating Program 1 Graphs (1-5) ===")

        # Graph 1: Winter vs Summer
        print("Graph 1: Winter vs Summer...")
        df1 = self.analyzer.winter_vs_summer()
        self.plot_line(
            df1, 'year',
            [('winter_avg', 'Winter'), ('summer_avg', 'Summer')],
            'Winter vs Summer Temperature Change', 'Year',
            'Temperature Change (°C)', 'graph_01_winter_vs_summer.png'
        )

        # Graph 2: Country Acceleration
        print("Graph 2: Country Acceleration...")
        df2 = self.analyzer.country_acceleration()
        self.plot_bar(
            df2.head(12), 'area', 'slope',
            'Top Countries by Temperature Acceleration',
            'Temperature Acceleration (°C/year)', 'Country',
            'graph_02_country_acceleration.png', horizontal=True
        )

        # Graph 3: Thermal Inertia
        print("Graph 3: Thermal Inertia...")
        df3 = self.analyzer.global_temperature()
        self.plot_line(
            df3, 'year', 'temp', 'Global Temperature Change',
            'Year', 'Temperature Change (°C)', 'graph_03_thermal_inertia.png'
        )

        # Graph 4: Monthly Warming
        print("Graph 4: Monthly Warming...")
        df4 = self.analyzer.monthly_warming()
        self.plot_bar(
            df4, 'month', 'slope', 'Monthly Warming Trends',
            'Month', 'Warming Rate (°C/year)', 'graph_04_monthly_warming.png'
        )

        # Graph 5: Projection
        print("Graph 5: Temperature Projections...")
        self._plot_projection(df3)

    def _plot_projection(self, df: pd.DataFrame) -> None:
        """Plot temperature threshold projections"""
        x, y = df['year'].values, df['temp'].values
        slope, intercept = np.polyfit(x, y, 1)
        years = np.arange(x.min(), x.max() + 100)

        plt.figure(figsize=FIGURE_SIZE)
        plt.plot(x, y, 'o', label='Observed', markersize=4)
        plt.plot(years, slope * years + intercept, '--',
                 label=f'Trend ({slope:.5f} °C/year)', linewidth=2)

        for thresh in TEMP_THRESHOLDS:
            year_cross = (thresh - intercept) / slope
            plt.axhline(thresh, linestyle=':', label=f'{thresh}°C Threshold')
            plt.plot(year_cross, thresh, 'X', markersize=12)
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
        """Generate graphs 6-10 with matching visual style"""
        print("\n=== Generating Program 2 Graphs (6-10) ===")

        df_long = self.analyzer.get_long_format_data()

        print("Graph 6: Global Temperature Over Years (Animated)...")
        self._plot_global_temp_animated(df_long)

        print("Graph 7: Top 30 Areas in 2019...")
        self._plot_top_areas(df_long, year=2019)

        print("Graph 8: China Temperature Changes...")
        self._plot_area_temp(df_long, area="China")

        print("Graph 9: Temperature Variability...")
        self._plot_temp_variability(df_long)

        print("Graph 10: Regional Comparison...")
        self._plot_regional_comparison(df_long)

    def _plot_global_temp_animated(self, df_long: pd.DataFrame) -> None:
        """Graph 6: Animated global temperature with glowing effect"""
        world_long = df_long[
            (df_long["Area"] == "World") &
            (df_long["Element"] == "Temperature change")
            ].copy()

        if world_long.empty:
            print("Warning: No World data found for animation")
            return

        pivot = world_long.pivot(
            index="Year", columns="Month_Num", values="Value"
        ).sort_index()

        years = pivot.index.values
        color_map = get_color_map(len(years))
        year_colors = [color_map(i / len(years)) for i in range(len(years))]

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(DARK_THEME["bg"])
        ax.set_facecolor(DARK_THEME["plot_bg"])

        def update(frame):
            ax.clear()

            ax.set_facecolor(DARK_THEME["plot_bg"])
            ax.set_xlim(0, 11)
            ax.set_ylim(-2, 3)
            ax.set_xticks(range(12))
            ax.set_xticklabels(MONTH_ABBREV, color=DARK_THEME["text"])
            ax.set_xlabel("Month", color=DARK_THEME["text"])
            ax.set_ylabel("Temperature Change (°C)", color=DARK_THEME["text"])
            ax.tick_params(colors=DARK_THEME["text"])

            current_year = years[frame]

            for i in range(frame):
                ax.plot(range(12), pivot.iloc[i],
                        color=year_colors[i], alpha=0.2, linewidth=1.5)

            bright_color = brighten_color(year_colors[frame])

            for width, alpha in [(11, 0.05), (9, 0.08), (7, 0.1)]:
                ax.plot(range(12), pivot.iloc[frame],
                        color=bright_color, linewidth=width, alpha=alpha)

            ax.plot(range(12), pivot.iloc[frame],
                    color=bright_color, linewidth=3.5, alpha=1.0)

            ax.text(0.5, 1.05, f"Global Temperature Change — {current_year}",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=16, weight="bold", color=DARK_THEME["text"])

        ani = FuncAnimation(fig, update, frames=len(years),
                            interval=400, repeat=True)

        plt.tight_layout()

        # Save animation as GIF
        try:
            ani.save('graph_06_global_temp_animated.gif', writer='pillow', fps=2)
            print("Saved: graph_06_global_temp_animated.gif")
        except Exception as e:
            print(f"Could not save animation: {e}")

        plt.show()
        plt.close()

    def _plot_top_areas(self, df_long: pd.DataFrame, year: int,
                        top_n: int = TOP_AREAS_COUNT) -> None:
        """Graph 7: Top areas by temperature change"""
        regions_long = df_long[df_long["Element"] == "Temperature change"].copy()
        df_year = regions_long[regions_long["Year"] == year]

        if df_year.empty:
            print(f"Warning: No data found for year {year}")
            return

        area_avg = df_year.groupby("Area")["Value"].mean().sort_values(
            ascending=False
        ).head(top_n)

        color_map = get_color_map()
        norm = Normalize(vmin=area_avg.min(), vmax=area_avg.max())
        bar_colors = [color_map(norm(val)) for val in area_avg.values]

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor(DARK_THEME["bg"])
        ax.set_facecolor(DARK_THEME["plot_bg"])
        ax.tick_params(colors=DARK_THEME["text"])
        ax.xaxis.label.set_color(DARK_THEME["text"])
        ax.yaxis.label.set_color(DARK_THEME["text"])
        ax.title.set_color(DARK_THEME["text"])

        ax.bar(area_avg.index, area_avg.values, color=bar_colors)
        ax.set_xticks(range(len(area_avg.index)))
        ax.set_xticklabels(area_avg.index, rotation=45, ha="right")
        ax.set_ylabel("Average Temperature Change (°C)")
        ax.set_title(f"Top {top_n} Areas with Highest Avg Temperature Change in {year}")

        plt.tight_layout()
        save_plot(f'graph_07_top30_areas_2019.png', dark_mode=True)
        plt.show()
        plt.close()

    def _plot_area_temp(self, df_long: pd.DataFrame, area: str) -> None:
        """Graph 8: Temperature changes for specific area"""
        regions_long = df_long[df_long["Element"] == "Temperature change"].copy()
        df_area = regions_long[regions_long["Area"] == area]

        if df_area.empty:
            print(f"Warning: No data found for area '{area}'")
            return

        pivot_area = df_area.pivot(
            index="Year", columns="Month_Num", values="Value"
        ).sort_index()

        years_area = pivot_area.index.values
        color_map_area = get_color_map(len(years_area))
        year_colors = [color_map_area(i / len(years_area))
                       for i in range(len(years_area))]

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor(DARK_THEME["bg"])
        ax.set_facecolor(DARK_THEME["plot_bg"])
        ax.tick_params(colors=DARK_THEME["text"])
        ax.xaxis.label.set_color(DARK_THEME["text"])
        ax.yaxis.label.set_color(DARK_THEME["text"])
        ax.title.set_color(DARK_THEME["text"])

        ax.set_xlim(0, 11)
        ax.set_xticks(range(12))
        ax.set_xticklabels(MONTH_ABBREV)
        ax.set_ylabel("Temperature Change (°C)")
        ax.set_xlabel("Month")
        ax.set_title(f"Monthly Temperature Changes in {area} Over the Years",
                     fontsize=14, weight="bold")

        for i in range(len(years_area)):
            ax.plot(range(12), pivot_area.iloc[i],
                    color=year_colors[i], linewidth=2, alpha=0.7)

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
        """Graph 9: Temperature variability over time"""
        std_data = df_long[
            (df_long["Element"] == "Standard Deviation") &
            (df_long["Area"] == "World")
            ].copy()

        if std_data.empty:
            print("Warning: No Standard Deviation data found for World")
            return

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

        for year in sorted(std_data["Year"].unique()):
            yearly_data = std_data[std_data["Year"] == year]
            ax.plot(yearly_data["Month_Num"], yearly_data["Value"],
                    color="lightsteelblue", linewidth=2.5)

        plt.tight_layout()
        save_plot('graph_09_temp_variability.png', dark_mode=True)
        plt.show()
        plt.close()

    def _plot_regional_comparison(self, df_long: pd.DataFrame) -> None:
        """Graph 10: Regional temperature comparison"""
        areas_to_compare = ["World", "Europe", "Asia", "Americas"]
        area_colors = {
            "World": "#a11216",
            "Europe": "#f89c56",
            "Asia": "#c5e7ad",
            "Americas": "#64aaaf"
        }

        regions_long = df_long[df_long["Element"] == "Temperature change"].copy()
        temp_compare = regions_long[regions_long["Area"].isin(areas_to_compare)]

        if temp_compare.empty:
            print("Warning: No data found for regional comparison")
            return

        area_yearly = temp_compare.groupby(["Area", "Year"])["Value"].mean().reset_index()

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

        for area in areas_to_compare:
            data = area_yearly[area_yearly["Area"] == area]
            ax.plot(data["Year"], data["Value"], label=area,
                    color=area_colors.get(area, "white"), linewidth=2)

        legend = ax.legend(facecolor=DARK_THEME["plot_bg"],
                           edgecolor=DARK_THEME["text"])
        for text in legend.get_texts():
            text.set_color(DARK_THEME["text"])

        plt.tight_layout()
        save_plot('graph_10_regional_comparison.png', dark_mode=True)
        plt.show()
        plt.close()

    def _generate_program3_graphs(self) -> None:
        """Generate graphs 11-15"""
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
            raise

    def _plot_afghanistan_temp(self) -> None:
        """Graph 11: Afghanistan temperature change by month"""
        df = self.analyzer.get_area_data('Afghanistan', year_start=1980, year_end=2000)

        if df.empty:
            print("Warning: No data found for Afghanistan (1980-2000)")
            return

        df = clean_month_data(df)
        df['Year'] = df['Year'].astype(int)

        plt.figure(figsize=FIGURE_SIZE)
        months = df['Months'].unique()
        colors_palette = cm.viridis(np.linspace(0, 1, len(months)))

        for i, month in enumerate(months):
            month_df = df[df['Months'] == month]
            plt.plot(month_df['Year'], month_df['Temperature'],
                     label=month, color=colors_palette[i], linewidth=2)

        plt.title("Average Temperature Change per Month in Afghanistan (1980–2000)")
        plt.xlabel("Year")
        plt.ylabel("Temperature Change (°C)")
        plt.legend(title="Month", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        save_plot('graph_11_afghanistan_temp.png')
        plt.show()
        plt.close()

    def _plot_month_variation(self) -> None:
        """Graph 12: Months with most temperature variation"""
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

        df = clean_month_data(df)

        df_std = df.groupby('Months')['Temperature'].std().reset_index()
        df_std.columns = ['Months', 'std']

        df_std['Months'] = pd.Categorical(df_std['Months'],
                                          categories=MONTHS, ordered=True)
        df_sorted = df_std.sort_values('std', ascending=False)

        plt.figure(figsize=FIGURE_SIZE)
        colors_palette = plt.cm.viridis(np.linspace(0, 1, len(df_sorted)))
        plt.bar(df_sorted['Months'].astype(str), df_sorted['std'],
                color=colors_palette)
        plt.title("Months with Most Temperature Variation")
        plt.xlabel("Month")
        plt.ylabel("Standard Deviation of Temperature (°C)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        save_plot('graph_12_month_variation.png')
        plt.show()
        plt.close()

    def _plot_seasonal_trend(self) -> None:
        """Graph 13: Seasonal temperature trends"""
        df = self.analyzer.get_area_data('Afghanistan', year_start=1961, year_end=2019)

        if df.empty:
            print("Warning: No data found for Afghanistan seasonal trends")
            return

        df = clean_month_data(df)
        df['Year'] = df['Year'].astype(int)

        month_to_season = {}
        for season, months in SEASONS.items():
            for month in months:
                month_to_season[month] = season

        df['Season'] = df['Months'].map(month_to_season)
        df_pivot = df.pivot_table(index='Year', columns='Season',
                                  values='Temperature', aggfunc='mean')

        df_pivot.plot.bar(stacked=True, figsize=FIGURE_SIZE, colormap=cm.viridis,
                          title="Average Seasonal Temperature per Year in Afghanistan")
        plt.xlabel("Year")
        plt.ylabel("Temperature Change (°C)")
        plt.tight_layout()
        save_plot('graph_13_seasonal_trend.png')
        plt.show()
        plt.close()

    def _plot_decade_trend(self) -> None:
        """Graph 14: Decade temperature trends"""
        df = self.analyzer.get_area_data('Afghanistan')

        if df.empty:
            print("Warning: No data found for Afghanistan decade trends")
            return

        df = clean_month_data(df)
        df['Year'] = df['Year'].astype(int)
        df = df[df['Year'].isin(DECADE_YEARS)]

        df_decade = df.groupby('Year')['Temperature'].mean()

        plt.figure(figsize=(8, 5))
        colors_palette = plt.cm.viridis(np.linspace(0, 1, len(df_decade)))
        bars = plt.bar(df_decade.index.astype(str), df_decade.values,
                       color=colors_palette, edgecolor='black')

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
        """Graph 15: Month comparison across all areas"""
        decades = [1961, 1970, 1980, 1990, 2000, 2010, 2019]
        query = f"""
            SELECT month as Months, year as Year, value as Temperature
            FROM temperature
            WHERE element='Temperature change'
            AND year IN ({','.join(map(str, decades))})
        """
        df = self.analyzer.db.query(query)
        df = clean_month_data(df)
        df['Year'] = df['Year'].astype(str)

        plt.figure(figsize=FIGURE_SIZE)
        palette = sns.color_palette("viridis", n_colors=len(df['Year'].unique()))
        sns.barplot(x='Months', y='Temperature', hue='Year', data=df, palette=palette)
        plt.title("Temperature Comparison Across Months")
        plt.xlabel("Month")
        plt.ylabel("Temperature Change (°C)")
        plt.legend(title="Year", bbox_to_anchor=(1.05, 1), loc='upper left')
        save_plot('graph_15_month_comparison_all.png')
        plt.show()
        plt.close()


# ==================== MAIN ====================
def main(source_table: str = 'temperature_data') -> None:
    """Main execution"""
    print("=" * 60)
    print("COMBINED TEMPERATURE ANALYSIS - 15 GRAPHS")
    print("=" * 60)

    try:
        db_manager = DatabaseManager(DB_CONFIG)

        print("\nSetting up temperature table...")
        db_manager.setup_temperature_table()

        print("Loading data from database table...")
        df = db_manager.load_and_transform_data(source_table)

        print("Importing to temperature table...")
        db_manager.import_data(df)

        print("\nStarting graph generation...")
        analyzer = TemperatureAnalyzer(db_manager)
        plot_generator = PlotGenerator(analyzer)
        plot_generator.generate_all_graphs()

    except ValueError as ve:
        print(f"\nValidation Error: {ve}")
        print("Please check your input parameters and try again.")
    except Exception as e:
        print(f"\nUnexpected Error: {e}")
        print("Please check your database connection and data.")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combined Temperature Analysis - 15 Graphs"
    )
    parser.add_argument(
        "--source",
        default="temperature_data",
        help=f"Source table name (allowed: {', '.join(ALLOWED_SOURCE_TABLES)})"
    )
    args = parser.parse_args()
    main(args.source)