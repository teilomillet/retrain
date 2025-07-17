"""
Performance optimization and monitoring for Retrain distributed training.

Provides runtime performance monitoring, optimization suggestions,
and automatic tuning capabilities.
"""

import logging
import time
from typing import Dict, Any, List
import statistics

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """
    Performance optimizer for Retrain distributed training.
    
    Monitors training performance and provides optimization recommendations
    for improved throughput and resource utilization.
    """
    
    def __init__(self, hardware_detector):
        """Initialize optimizer with hardware detection results."""
        self.detector = hardware_detector
        self.capabilities = hardware_detector.capabilities
        self.recommendations = hardware_detector.recommendations
        
        # Performance tracking
        self.metrics_history = []
        self.optimization_history = []
        self.current_config = {}
        
        # Performance thresholds
        self.thresholds = {
            'episode_time_warning': 60.0,  # seconds
            'episode_time_critical': 120.0,  # seconds
            'memory_warning': 85.0,  # percent
            'memory_critical': 95.0,  # percent
            'gpu_util_low': 30.0,  # percent
            'gpu_util_high': 95.0,  # percent
            'cpu_util_low': 20.0,  # percent
            'cpu_util_high': 90.0,  # percent
        }
        
    def record_episode_metrics(self, episode_metrics: Dict[str, Any]) -> None:
        """Record performance metrics from a training episode."""
        enriched_metrics = {
            **episode_metrics,
            'timestamp': time.time(),
            'hardware_context': {
                'device_type': self.capabilities['device']['primary_device'],
                'device_count': self.capabilities['device']['device_count'],
                'deployment_type': self.recommendations['deployment_type'],
                'backend': self.recommendations['backend']
            }
        }
        
        self.metrics_history.append(enriched_metrics)
        
        # Keep only recent history (last 100 episodes)
        if len(self.metrics_history) > 100:
            self.metrics_history.pop(0)
            
        # Analyze performance and suggest optimizations
        if len(self.metrics_history) >= 5:
            self._analyze_performance_trends()
            
    def _analyze_performance_trends(self) -> None:
        """Analyze recent performance trends and identify issues."""
        recent_metrics = self.metrics_history[-10:]  # Last 10 episodes
        
        # Extract episode times
        episode_times = [m.get('episode_time', 0) for m in recent_metrics if 'episode_time' in m]
        
        if not episode_times:
            return
            
        avg_time = statistics.mean(episode_times)
        time_trend = self._calculate_trend(episode_times)
        
        # Analyze training metrics if available
        training_losses = []
        for m in recent_metrics:
            if 'training_metrics' in m and isinstance(m['training_metrics'], dict):
                loss = m['training_metrics'].get('total_loss', m['training_metrics'].get('loss'))
                if loss is not None:
                    training_losses.append(loss)
                    
        # Record analysis
        analysis = {
            'timestamp': time.time(),
            'avg_episode_time': avg_time,
            'episode_time_trend': time_trend,
            'avg_training_loss': statistics.mean(training_losses) if training_losses else None,
            'loss_trend': self._calculate_trend(training_losses) if len(training_losses) >= 3 else None,
            'performance_status': self._assess_performance_status(avg_time, time_trend)
        }
        
        self.optimization_history.append(analysis)
        
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a list of values."""
        if len(values) < 3:
            return "insufficient_data"
            
        # Simple linear trend analysis
        x = list(range(len(values)))
        y = values
        
        # Calculate slope
        n = len(values)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        
        if abs(slope) < 0.01:  # Threshold for "stable"
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"
            
    def _assess_performance_status(self, avg_time: float, trend: str) -> str:
        """Assess overall performance status."""
        if avg_time > self.thresholds['episode_time_critical']:
            return "critical"
        elif avg_time > self.thresholds['episode_time_warning']:
            return "warning"
        elif trend == "increasing":
            return "degrading"
        elif trend == "decreasing":
            return "improving"
        else:
            return "stable"
            
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get specific optimization recommendations based on analysis."""
        if not self.optimization_history:
            return [{"type": "info", "message": "Need more performance data for recommendations"}]
            
        recommendations = []
        latest_analysis = self.optimization_history[-1]
        
        # Episode time recommendations
        avg_time = latest_analysis['avg_episode_time']
        time_trend = latest_analysis['episode_time_trend']
        
        if avg_time > self.thresholds['episode_time_critical']:
            recommendations.append({
                "type": "critical",
                "category": "performance",
                "message": f"Very slow episodes ({avg_time:.1f}s avg)",
                "suggestions": [
                    "Reduce batch size significantly",
                    "Use smaller model or quantization",
                    "Check for CPU/GPU bottlenecks",
                    "Consider distributed training across more nodes"
                ]
            })
        elif avg_time > self.thresholds['episode_time_warning']:
            recommendations.append({
                "type": "warning",
                "category": "performance",
                "message": f"Slow episodes ({avg_time:.1f}s avg)",
                "suggestions": [
                    "Reduce batch size",
                    "Optimize data loading pipeline",
                    "Check memory usage patterns",
                    "Consider gradient accumulation"
                ]
            })
            
        # Trend-based recommendations
        if time_trend == "increasing":
            recommendations.append({
                "type": "warning",
                "category": "trend",
                "message": "Episode times are increasing over time",
                "suggestions": [
                    "Check for memory leaks",
                    "Monitor GPU memory fragmentation",
                    "Review data pipeline efficiency",
                    "Consider periodic model checkpointing"
                ]
            })
            
        # Hardware-specific recommendations
        device_type = self.capabilities['device']['primary_device']
        
        if device_type == 'cpu':
            recommendations.append({
                "type": "info",
                "category": "hardware",
                "message": "CPU-only training detected",
                "suggestions": [
                    "Use smaller models for faster iteration",
                    "Enable multiprocessing for data loading",
                    "Consider mixed precision if supported",
                    "Reduce sequence length to improve throughput"
                ]
            })
        elif device_type == 'mps':
            recommendations.append({
                "type": "info", 
                "category": "hardware",
                "message": "Apple Silicon MPS training",
                "suggestions": [
                    "Use fp16 precision for better performance",
                    "Monitor unified memory usage",
                    "Batch size 2-4 often optimal for MPS",
                    "Avoid very large models due to memory constraints"
                ]
            })
            
        # Backend-specific recommendations
        backend = self.recommendations['backend']
        
        if backend == 'transformers':
            recommendations.append({
                "type": "info",
                "category": "backend",
                "message": "Using HuggingFace Transformers backend",
                "suggestions": [
                    "Enable gradient checkpointing for memory efficiency",
                    "Use DataLoader with multiple workers",
                    "Consider torch.compile() for PyTorch 2.0+",
                    "Enable mixed precision training"
                ]
            })
        elif backend == 'mbridge':
            recommendations.append({
                "type": "info",
                "category": "backend", 
                "message": "Using MBridge backend for distributed training",
                "suggestions": [
                    "Optimize tensor/pipeline parallelism settings",
                    "Monitor NCCL communication efficiency",
                    "Balance load across GPU workers",
                    "Use appropriate micro-batch sizes"
                ]
            })
            
        return recommendations if recommendations else [
            {"type": "info", "message": "Performance looks good! No specific optimizations needed."}
        ]
        
    def suggest_config_adjustments(self) -> Dict[str, Any]:
        """Suggest specific configuration adjustments."""
        adjustments = {
            'batch_size': None,
            'learning_rate': None,
            'sequence_length': None,
            'gradient_accumulation': None,
            'checkpoint_frequency': None,
            'reasoning': []
        }
        
        if not self.metrics_history:
            return adjustments
            
        recent_metrics = self.metrics_history[-5:]
        avg_time = statistics.mean([m.get('episode_time', 0) for m in recent_metrics])
        
        # Batch size adjustments
        current_batch_size = self.current_config.get('batch_size', 4)
        
        if avg_time > self.thresholds['episode_time_warning']:
            # Reduce batch size for faster episodes
            new_batch_size = max(1, current_batch_size // 2)
            adjustments['batch_size'] = new_batch_size
            adjustments['reasoning'].append(f"Reduce batch size from {current_batch_size} to {new_batch_size} for faster episodes")
            
        elif avg_time < 10.0 and current_batch_size < 16:
            # Increase batch size for better efficiency
            new_batch_size = min(16, current_batch_size * 2)
            adjustments['batch_size'] = new_batch_size
            adjustments['reasoning'].append(f"Increase batch size from {current_batch_size} to {new_batch_size} for better GPU utilization")
            
        # Gradient accumulation adjustments
        if adjustments['batch_size'] and adjustments['batch_size'] < current_batch_size:
            # Compensate reduced batch size with gradient accumulation
            accumulation_steps = current_batch_size // adjustments['batch_size']
            adjustments['gradient_accumulation'] = accumulation_steps
            adjustments['reasoning'].append(f"Use gradient accumulation steps of {accumulation_steps} to maintain effective batch size")
            
        # Checkpoint frequency adjustments
        if avg_time > 30.0:
            # More frequent checkpointing for long episodes
            adjustments['checkpoint_frequency'] = 5
            adjustments['reasoning'].append("Increase checkpoint frequency for long episodes")
        elif avg_time < 5.0:
            # Less frequent checkpointing for fast episodes
            adjustments['checkpoint_frequency'] = 20
            adjustments['reasoning'].append("Reduce checkpoint frequency for fast episodes")
            
        return adjustments
        
    def monitor_realtime_performance(self) -> Dict[str, Any]:
        """Monitor real-time performance metrics."""
        try:
            import psutil
            import torch
            
            metrics = {
                'timestamp': time.time(),
                'system': {
                    'cpu_percent': psutil.cpu_percent(interval=1),
                    'memory_percent': psutil.virtual_memory().percent,
                    'memory_available_gb': psutil.virtual_memory().available / 1e9
                },
                'torch': {
                    'cuda_available': torch.cuda.is_available(),
                    'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
                },
                'status': 'healthy'
            }
            
            # GPU metrics if available
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1e9
                gpu_reserved = torch.cuda.memory_reserved() / 1e9
                gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                
                metrics['gpu'] = {
                    'memory_allocated_gb': gpu_memory,
                    'memory_reserved_gb': gpu_reserved,
                    'memory_total_gb': gpu_total,
                    'memory_percent': (gpu_reserved / gpu_total) * 100,
                    'device_name': torch.cuda.get_device_name(0)
                }
                
                # Status warnings
                if metrics['gpu']['memory_percent'] > self.thresholds['memory_critical']:
                    metrics['status'] = 'critical'
                    metrics['warnings'] = ['Critical GPU memory usage']
                elif metrics['gpu']['memory_percent'] > self.thresholds['memory_warning']:
                    metrics['status'] = 'warning'
                    metrics['warnings'] = ['High GPU memory usage']
                    
            # System status warnings
            if metrics['system']['memory_percent'] > self.thresholds['memory_critical']:
                metrics['status'] = 'critical'
                metrics.setdefault('warnings', []).append('Critical system memory usage')
            elif metrics['system']['cpu_percent'] > self.thresholds['cpu_util_high']:
                metrics['status'] = 'warning'
                metrics.setdefault('warnings', []).append('High CPU usage')
                
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to monitor real-time performance: {e}")
            return {'error': str(e), 'timestamp': time.time()}
            
    def print_performance_report(self) -> None:
        """Print comprehensive performance analysis report."""
        print("\n📈 Performance Analysis Report")
        print("=" * 50)
        
        if not self.metrics_history:
            print("No performance data available yet.")
            return
            
        # Recent performance summary
        recent_metrics = self.metrics_history[-10:]
        episode_times = [m.get('episode_time', 0) for m in recent_metrics if 'episode_time' in m]
        
        if episode_times:
            print(f"Recent Performance (last {len(episode_times)} episodes):")
            print(f"  • Average episode time: {statistics.mean(episode_times):.2f}s")
            print(f"  • Fastest episode: {min(episode_times):.2f}s")
            print(f"  • Slowest episode: {max(episode_times):.2f}s")
            print(f"  • Time variance: {statistics.variance(episode_times):.2f}")
            
        # Performance status
        if self.optimization_history:
            latest_analysis = self.optimization_history[-1]
            status = latest_analysis['performance_status']
            trend = latest_analysis['episode_time_trend']
            
            status_emoji = {
                'critical': '🔴',
                'warning': '🟡',
                'degrading': '📉',
                'improving': '📈',
                'stable': '✅'
            }.get(status, '❓')
            
            print(f"\nPerformance Status: {status_emoji} {status.title()}")
            print(f"Trend: {trend.title()}")
            
        # Optimization recommendations
        recommendations = self.get_optimization_recommendations()
        if recommendations:
            print(  "\n💡 Recommendations:")
            for rec in recommendations:
                rec_type = rec.get('type', 'info')
                category = rec.get('category', 'general')
                message = rec.get('message', '')
                
                emoji = {'critical': '🚨', 'warning': '⚠️', 'info': 'ℹ️'}.get(rec_type, 'ℹ️')
                print(f"  {emoji} [{category.title()}] {message}")
                
                if 'suggestions' in rec:
                    for suggestion in rec['suggestions']:
                        print(f"    → {suggestion}")
                        
        # Real-time metrics
        realtime = self.monitor_realtime_performance()
        if 'error' not in realtime:
            print("\nCurrent System Status:")
            print(f"  • CPU Usage: {realtime['system']['cpu_percent']:.1f}%")
            print(f"  • RAM Usage: {realtime['system']['memory_percent']:.1f}%")
            print(f"  • Available RAM: {realtime['system']['memory_available_gb']:.1f}GB")
            
            if 'gpu' in realtime:
                print(f"  • GPU Memory: {realtime['gpu']['memory_percent']:.1f}%")
                print(f"  • GPU Device: {realtime['gpu']['device_name']}")
                
        # Hardware context
        print("\nHardware Context:")
        print(f"  • Platform: {self.capabilities['platform']['system']}")
        print(f"  • Device: {self.capabilities['device']['primary_device']}")
        print(f"  • Backend: {self.recommendations['backend']}")
        print(f"  • Deployment: {self.recommendations['deployment_type']}")
        
    def update_config(self, config: Dict[str, Any]) -> None:
        """Update current configuration for optimization tracking."""
        self.current_config.update(config)
        
    def get_performance_score(self) -> float:
        """Calculate overall performance score (0-100)."""
        if not self.metrics_history:
            return 50.0  # Neutral score
            
        recent_metrics = self.metrics_history[-5:]
        episode_times = [m.get('episode_time', 30) for m in recent_metrics if 'episode_time' in m]
        
        if not episode_times:
            return 50.0
            
        avg_time = statistics.mean(episode_times)
        
        # Score based on episode time relative to thresholds
        if avg_time <= 5.0:
            return 100.0  # Excellent
        elif avg_time <= 15.0:
            return 80.0   # Good
        elif avg_time <= 30.0:
            return 60.0   # Fair
        elif avg_time <= 60.0:
            return 40.0   # Poor
        else:
            return 20.0   # Very poor 