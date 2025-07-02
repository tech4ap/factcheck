#!/usr/bin/env python3
"""
Comprehensive visualization and analysis script for deepfake detection results.

This script provides detailed analysis and visualization of:
- Training history and performance
- Model evaluation results
- Data distribution analysis
- Confusion matrices and ROC curves
- Model comparison charts
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import VIDEO_TRAIN_DIR, VIDEO_VAL_DIR, VIDEO_TEST_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure matplotlib for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ResultsAnalyzer:
    """
    Comprehensive results analyzer for deepfake detection models.
    """
    
    def __init__(self, models_dir: str = "models", output_dir: str = "analysis_results"):
        """
        Initialize the results analyzer.
        
        Args:
            models_dir: Directory containing model files and results
            output_dir: Directory to save analysis results
        """
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "data_analysis").mkdir(exist_ok=True)
        
        logger.info(f"Initialized ResultsAnalyzer with models_dir: {self.models_dir}")
    
    def analyze_data_distribution(self) -> Dict[str, Dict[str, int]]:
        """
        Analyze the distribution of data across splits.
        
        Returns:
            Dictionary containing data counts for each split
        """
        logger.info("Analyzing data distribution...")
        
        distribution = {}
        
        # Analyze video data
        try:
            train_fake = len(list(Path(VIDEO_TRAIN_DIR).glob("fake/*.mp4")))
            train_real = len(list(Path(VIDEO_TRAIN_DIR).glob("real/*.mp4")))
            val_fake = len(list(Path(VIDEO_VAL_DIR).glob("fake/*.mp4")))
            val_real = len(list(Path(VIDEO_VAL_DIR).glob("real/*.mp4")))
            test_fake = len(list(Path(VIDEO_TEST_DIR).glob("fake/*.mp4")))
            test_real = len(list(Path(VIDEO_TEST_DIR).glob("real/*.mp4")))
            
            distribution['video'] = {
                'train': {'fake': train_fake, 'real': train_real},
                'validation': {'fake': val_fake, 'real': val_real},
                'test': {'fake': test_fake, 'real': test_real}
            }
            
            logger.info(f"Video data distribution: {distribution['video']}")
            
        except Exception as e:
            logger.warning(f"Error analyzing video data distribution: {e}")
        
        return distribution
    
    def create_data_distribution_plots(self, distribution: Dict[str, Dict[str, int]]) -> None:
        """
        Create comprehensive data distribution visualizations.
        
        Args:
            distribution: Data distribution dictionary
        """
        logger.info("Creating data distribution plots...")
        
        for media_type, splits in distribution.items():
            # Prepare data for plotting
            split_names = list(splits.keys())
            fake_counts = [splits[split]['fake'] for split in split_names]
            real_counts = [splits[split]['real'] for split in split_names]
            
            # Create comprehensive visualization
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{media_type.title()} Data Distribution Analysis', fontsize=16, fontweight='bold')
            
            # Stacked bar chart
            x = np.arange(len(split_names))
            width = 0.35
            
            ax1.bar(x, real_counts, width, label='Real', color='skyblue', alpha=0.8)
            ax1.bar(x, fake_counts, width, bottom=real_counts, label='Fake', color='lightcoral', alpha=0.8)
            
            ax1.set_xlabel('Data Split')
            ax1.set_ylabel('Number of Samples')
            ax1.set_title('Sample Distribution by Split')
            ax1.set_xticks(x)
            ax1.set_xticklabels(split_names)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Add value labels
            for i, (real, fake) in enumerate(zip(real_counts, fake_counts)):
                ax1.text(i, real/2, str(real), ha='center', va='center', fontweight='bold')
                ax1.text(i, real + fake/2, str(fake), ha='center', va='center', fontweight='bold')
            
            # Pie chart for total distribution
            total_real = sum(real_counts)
            total_fake = sum(fake_counts)
            
            ax2.pie([total_real, total_fake], labels=['Real', 'Fake'], 
                   autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral'])
            ax2.set_title('Overall Data Distribution')
            
            # Bar chart comparing fake vs real
            ax3.bar(['Real', 'Fake'], [total_real, total_fake], 
                   color=['skyblue', 'lightcoral'], alpha=0.8)
            ax3.set_ylabel('Number of Samples')
            ax3.set_title('Total Samples by Class')
            ax3.grid(True, alpha=0.3)
            
            # Add value labels
            for i, count in enumerate([total_real, total_fake]):
                ax3.text(i, count + max([total_real, total_fake]) * 0.01, str(count), 
                        ha='center', va='bottom', fontweight='bold')
            
            # Split-wise comparison
            ax4.bar(np.arange(len(split_names)) - width/2, real_counts, width, 
                   label='Real', color='skyblue', alpha=0.8)
            ax4.bar(np.arange(len(split_names)) + width/2, fake_counts, width, 
                   label='Fake', color='lightcoral', alpha=0.8)
            ax4.set_xlabel('Data Split')
            ax4.set_ylabel('Number of Samples')
            ax4.set_title('Split-wise Class Distribution')
            ax4.set_xticks(np.arange(len(split_names)))
            ax4.set_xticklabels(split_names)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "data_analysis" / f"{media_type}_data_distribution.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Created data distribution plots for {media_type}")
    
    def analyze_training_results(self) -> Dict[str, Dict]:
        """
        Analyze training results from CSV files.
        
        Returns:
            Dictionary containing training analysis results
        """
        logger.info("Analyzing training results...")
        
        results = {}
        
        # Load evaluation results
        eval_csv = self.models_dir / "model_evaluation_results.csv"
        if eval_csv.exists():
            df = pd.read_csv(eval_csv, index_col=0)
            results['evaluation'] = df.to_dict('index')
            logger.info(f"Loaded evaluation results: {results['evaluation']}")
        
        # Check for training history plots
        training_plots_dir = self.models_dir / "training_plots"
        if training_plots_dir.exists():
            plot_files = list(training_plots_dir.glob("*.png"))
            results['training_plots'] = [str(f) for f in plot_files]
            logger.info(f"Found {len(plot_files)} training plot files")
        
        # Check for performance plots
        performance_plots_dir = self.models_dir / "performance_plots"
        if performance_plots_dir.exists():
            plot_files = list(performance_plots_dir.glob("*.png"))
            results['performance_plots'] = [str(f) for f in plot_files]
            logger.info(f"Found {len(plot_files)} performance plot files")
        
        return results
    
    def create_performance_summary(self, results: Dict[str, Dict]) -> None:
        """
        Create a comprehensive performance summary report.
        
        Args:
            results: Analysis results dictionary
        """
        logger.info("Creating performance summary report...")
        
        # Create summary figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Deepfake Detection Model Performance Summary', fontsize=16, fontweight='bold')
        
        if 'evaluation' in results:
            eval_data = results['evaluation']
            
            # Extract metrics for plotting
            models = list(eval_data.keys())
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
            
            # Performance comparison
            for i, metric in enumerate(metrics[:4]):  # Plot first 4 metrics
                row = i // 2
                col = i % 2
                
                values = [eval_data[model].get(metric, 0) for model in models]
                
                bars = axes[row, col].bar(models, values, 
                                        color=['skyblue', 'lightcoral', 'lightgreen'][:len(models)])
                axes[row, col].set_title(f'{metric.replace("_", " ").title()}')
                axes[row, col].set_ylabel(metric.replace("_", " ").title())
                axes[row, col].set_ylim(0, 1)
                axes[row, col].grid(True, alpha=0.3)
                
                # Add value labels
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    axes[row, col].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                      f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Add summary text
        summary_text = "Performance Summary:\n\n"
        if 'evaluation' in results:
            for model, metrics in results['evaluation'].items():
                summary_text += f"{model.title()} Model:\n"
                for metric, value in metrics.items():
                    summary_text += f"  {metric}: {value:.3f}\n"
                summary_text += "\n"
        
        # Add recommendations
        summary_text += "Recommendations:\n"
        summary_text += "- Consider data augmentation for better generalization\n"
        summary_text += "- Try different model architectures\n"
        summary_text += "- Increase training data if possible\n"
        summary_text += "- Experiment with different learning rates\n"
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_title('Summary & Recommendations')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "performance_summary.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Created performance summary report")
    
    def generate_html_report(self, distribution: Dict[str, Dict[str, int]], 
                           results: Dict[str, Dict]) -> None:
        """
        Generate an HTML report with all analysis results.
        
        Args:
            distribution: Data distribution analysis
            results: Training results analysis
        """
        logger.info("Generating HTML report...")
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Deepfake Detection Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #333; }}
                .section {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f5f5f5; border-radius: 3px; }}
                .plot {{ text-align: center; margin: 20px 0; }}
                .plot img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Deepfake Detection Analysis Report</h1>
            <p>Generated on: {timestamp}</p>
        """.format(timestamp=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Data Distribution Section
        html_content += """
            <div class="section">
                <h2>Data Distribution Analysis</h2>
        """
        
        for media_type, splits in distribution.items():
            html_content += f"<h3>{media_type.title()} Data</h3>"
            html_content += "<table><tr><th>Split</th><th>Real</th><th>Fake</th><th>Total</th></tr>"
            
            for split_name, counts in splits.items():
                total = counts['real'] + counts['fake']
                html_content += f"<tr><td>{split_name}</td><td>{counts['real']}</td><td>{counts['fake']}</td><td>{total}</td></tr>"
            
            html_content += "</table>"
        
        html_content += "</div>"
        
        # Performance Results Section
        if 'evaluation' in results:
            html_content += """
                <div class="section">
                    <h2>Model Performance Results</h2>
            """
            
            for model, metrics in results['evaluation'].items():
                html_content += f"<h3>{model.title()} Model</h3>"
                for metric, value in metrics.items():
                    html_content += f'<div class="metric"><strong>{metric}:</strong> {value:.3f}</div>'
            
            html_content += "</div>"
        
        # Plots Section
        html_content += """
            <div class="section">
                <h2>Visualizations</h2>
        """
        
        # Add data distribution plots
        for media_type in distribution.keys():
            plot_path = f"data_analysis/{media_type}_data_distribution.png"
            if (self.output_dir / plot_path).exists():
                html_content += f"""
                    <div class="plot">
                        <h3>{media_type.title()} Data Distribution</h3>
                        <img src="{plot_path}" alt="{media_type} data distribution">
                    </div>
                """
        
        # Add performance summary
        if (self.output_dir / "plots" / "performance_summary.png").exists():
            html_content += """
                <div class="plot">
                    <h3>Performance Summary</h3>
                    <img src="plots/performance_summary.png" alt="Performance summary">
                </div>
            """
        
        html_content += "</div>"
        
        # Recommendations Section
        html_content += """
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
                    <li><strong>Data Quality:</strong> Ensure balanced representation of real and fake samples</li>
                    <li><strong>Model Architecture:</strong> Consider ensemble methods for improved performance</li>
                    <li><strong>Training Strategy:</strong> Implement learning rate scheduling and early stopping</li>
                    <li><strong>Evaluation:</strong> Use cross-validation for more robust performance assessment</li>
                    <li><strong>Deployment:</strong> Monitor model performance on new data</li>
                </ul>
            </div>
        """
        
        html_content += """
            </body>
        </html>
        """
        
        # Save HTML report
        with open(self.output_dir / "reports" / "analysis_report.html", 'w') as f:
            f.write(html_content)
        
        logger.info(f"Generated HTML report: {self.output_dir / 'reports' / 'analysis_report.html'}")
    
    def run_complete_analysis(self) -> None:
        """
        Run the complete analysis pipeline.
        """
        logger.info("Starting complete analysis pipeline...")
        
        # Analyze data distribution
        distribution = self.analyze_data_distribution()
        
        # Create data distribution plots
        self.create_data_distribution_plots(distribution)
        
        # Analyze training results
        results = self.analyze_training_results()
        
        # Create performance summary
        self.create_performance_summary(results)
        
        # Generate HTML report
        self.generate_html_report(distribution, results)
        
        # Save analysis results as JSON
        analysis_results = {
            'data_distribution': distribution,
            'training_results': results,
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(self.output_dir / "reports" / "analysis_results.json", 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        logger.info("Complete analysis pipeline finished!")
        logger.info(f"Results saved to: {self.output_dir}")


def main():
    """Main function to run the analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze deepfake detection results")
    parser.add_argument("--models-dir", default="models", help="Directory containing model files")
    parser.add_argument("--output-dir", default="analysis_results", help="Output directory for analysis")
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = ResultsAnalyzer(args.models_dir, args.output_dir)
    analyzer.run_complete_analysis()
    
    print(f"\n✅ Analysis complete! Results saved to: {args.output_dir}")
    print(f"📊 View the HTML report: {args.output_dir}/reports/analysis_report.html")


if __name__ == "__main__":
    main() 