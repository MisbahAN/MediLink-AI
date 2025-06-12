"""
Concurrent Processing service for parallel document processing operations.

This service manages async processing of multiple PDF pages, batch operations,
and worker pool management with progress tracking and error handling for
efficient medical document processing.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from ..core.config import get_settings
from ..models.schemas import ProcessingStatusEnum

logger = logging.getLogger(__name__)
settings = get_settings()


class ProcessingStage(str, Enum):
    """Stages of concurrent processing operations."""
    INITIALIZING = "initializing"
    CHUNKING = "chunking"
    EXTRACTING = "extracting"
    MAPPING = "mapping"
    VALIDATING = "validating"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProcessingTask:
    """Individual processing task definition."""
    task_id: str
    task_type: str
    input_data: Any
    priority: int = 1
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)


@dataclass
class ProcessingResult:
    """Result of a processing task."""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    processing_time: float = 0.0
    completed_at: datetime = None
    
    def __post_init__(self):
        if self.completed_at is None:
            self.completed_at = datetime.now(timezone.utc)


@dataclass
class ProgressState:
    """Progress tracking state for concurrent operations."""
    session_id: str
    total_tasks: int
    completed_tasks: int = 0
    failed_tasks: int = 0
    current_stage: ProcessingStage = ProcessingStage.INITIALIZING
    stage_progress: float = 0.0
    start_time: datetime = None
    estimated_completion: Optional[datetime] = None
    error_messages: List[str] = None
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now(timezone.utc)
        if self.error_messages is None:
            self.error_messages = []
    
    @property
    def overall_progress(self) -> float:
        """Calculate overall progress percentage."""
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks / self.total_tasks) * 100.0
    
    @property
    def is_completed(self) -> bool:
        """Check if all tasks are completed."""
        return self.completed_tasks + self.failed_tasks >= self.total_tasks


class AsyncProcessor:
    """
    Advanced concurrent processing engine for medical document operations.
    
    Manages parallel processing of PDF pages, batch operations, and provides
    comprehensive progress tracking with error handling and recovery.
    """
    
    def __init__(self):
        """Initialize the async processor with configuration."""
        self.max_workers = settings.CONCURRENT_PROCESSING_LIMIT
        self.default_timeout = settings.PROCESSING_TIMEOUT_SECONDS
        
        # Processing state tracking
        self.active_sessions: Dict[str, ProgressState] = {}
        self.task_results: Dict[str, List[ProcessingResult]] = {}
        
        # Worker pool configuration
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.semaphore = asyncio.Semaphore(self.max_workers)
        
        # Performance tracking
        self.performance_metrics = {
            "total_tasks_processed": 0,
            "average_task_time": 0.0,
            "success_rate": 1.0,
            "peak_concurrent_tasks": 0
        }
        
        logger.info(f"Async processor initialized with {self.max_workers} max workers")
    
    async def process_pages_concurrently(
        self,
        session_id: str,
        pages_data: List[Dict[str, Any]],
        processing_function: Callable,
        batch_size: Optional[int] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Process multiple PDF pages concurrently with progress tracking.
        
        Args:
            session_id: Unique session identifier
            pages_data: List of page data to process
            processing_function: Async function to process each page
            batch_size: Optional batch size for processing (defaults to max_workers)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing processing results and statistics
        """
        if not pages_data:
            logger.warning(f"No pages to process for session {session_id}")
            return {"results": [], "success": True, "total_processed": 0}
        
        # Initialize progress tracking
        progress_state = ProgressState(
            session_id=session_id,
            total_tasks=len(pages_data),
            current_stage=ProcessingStage.INITIALIZING
        )
        self.active_sessions[session_id] = progress_state
        self.task_results[session_id] = []
        
        try:
            batch_size = batch_size or self.max_workers
            
            # Update progress to chunking stage
            progress_state.current_stage = ProcessingStage.CHUNKING
            if progress_callback:
                await progress_callback(progress_state)
            
            # Create processing tasks
            tasks = []
            for i, page_data in enumerate(pages_data):
                task = ProcessingTask(
                    task_id=f"{session_id}_page_{i}",
                    task_type="page_processing",
                    input_data=page_data,
                    priority=1,
                    timeout=self.default_timeout
                )
                tasks.append(task)
            
            # Process in batches
            progress_state.current_stage = ProcessingStage.EXTRACTING
            all_results = []
            
            for batch_start in range(0, len(tasks), batch_size):
                batch_end = min(batch_start + batch_size, len(tasks))
                batch_tasks = tasks[batch_start:batch_end]
                
                logger.info(f"Processing batch {batch_start // batch_size + 1}: "
                          f"pages {batch_start}-{batch_end-1}")
                
                # Process batch concurrently
                batch_results = await self._process_task_batch(
                    batch_tasks, processing_function, progress_state, progress_callback
                )
                
                all_results.extend(batch_results)
                
                # Small delay between batches to prevent overwhelming
                if batch_end < len(tasks):
                    await asyncio.sleep(0.1)
            
            # Finalize results
            progress_state.current_stage = ProcessingStage.FINALIZING
            if progress_callback:
                await progress_callback(progress_state)
            
            # Calculate statistics
            successful_results = [r for r in all_results if r.success]
            failed_results = [r for r in all_results if not r.success]
            
            processing_stats = self._calculate_processing_stats(
                all_results, progress_state.start_time
            )
            
            # Mark as completed
            progress_state.current_stage = ProcessingStage.COMPLETED
            if progress_callback:
                await progress_callback(progress_state)
            
            logger.info(f"Concurrent processing completed for session {session_id}: "
                       f"{len(successful_results)} successful, {len(failed_results)} failed")
            
            return {
                "session_id": session_id,
                "success": True,
                "results": [r.result for r in successful_results],
                "failed_results": [{"task_id": r.task_id, "error": r.error} for r in failed_results],
                "statistics": processing_stats,
                "total_processed": len(all_results),
                "success_count": len(successful_results),
                "failure_count": len(failed_results)
            }
            
        except Exception as e:
            logger.error(f"Concurrent processing failed for session {session_id}: {e}")
            progress_state.current_stage = ProcessingStage.FAILED
            progress_state.error_messages.append(str(e))
            
            if progress_callback:
                await progress_callback(progress_state)
            
            return {
                "session_id": session_id,
                "success": False,
                "error": str(e),
                "partial_results": self.task_results.get(session_id, []),
                "statistics": {"error": "Processing failed"}
            }
        
        finally:
            # Cleanup session tracking
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
    
    async def create_worker_pool(
        self,
        pool_size: Optional[int] = None,
        thread_pool: bool = False
    ) -> Union[ThreadPoolExecutor, None]:
        """
        Create and configure worker pool for processing operations.
        
        Args:
            pool_size: Size of the worker pool (defaults to max_workers)
            thread_pool: Whether to create a thread pool (for CPU-bound tasks)
            
        Returns:
            ThreadPoolExecutor if thread_pool=True, None otherwise
        """
        pool_size = pool_size or self.max_workers
        
        if thread_pool:
            if self.thread_pool:
                self.thread_pool.shutdown(wait=False)
            
            self.thread_pool = ThreadPoolExecutor(
                max_workers=pool_size,
                thread_name_prefix="medical_doc_processor"
            )
            
            logger.info(f"Thread pool created with {pool_size} workers")
            return self.thread_pool
        else:
            # For async operations, we use semaphore-based limiting
            self.semaphore = asyncio.Semaphore(pool_size)
            logger.info(f"Async semaphore configured for {pool_size} concurrent operations")
            return None
    
    async def _process_task_batch(
        self,
        batch_tasks: List[ProcessingTask],
        processing_function: Callable,
        progress_state: ProgressState,
        progress_callback: Optional[Callable] = None
    ) -> List[ProcessingResult]:
        """
        Process a batch of tasks concurrently.
        
        Args:
            batch_tasks: List of tasks to process
            processing_function: Function to process each task
            progress_state: Progress tracking state
            progress_callback: Optional progress callback
            
        Returns:
            List of processing results
        """
        async def process_single_task(task: ProcessingTask) -> ProcessingResult:
            """Process a single task with error handling and retry logic."""
            async with self.semaphore:  # Limit concurrent operations
                start_time = time.time()
                
                for attempt in range(task.max_retries + 1):
                    try:
                        # Apply timeout if specified
                        if task.timeout:
                            result = await asyncio.wait_for(
                                processing_function(task.input_data),
                                timeout=task.timeout
                            )
                        else:
                            result = await processing_function(task.input_data)
                        
                        processing_time = time.time() - start_time
                        
                        # Update performance metrics
                        self.performance_metrics["total_tasks_processed"] += 1
                        
                        # Update progress
                        progress_state.completed_tasks += 1
                        if progress_callback:
                            await progress_callback(progress_state)
                        
                        return ProcessingResult(
                            task_id=task.task_id,
                            success=True,
                            result=result,
                            processing_time=processing_time
                        )
                        
                    except asyncio.TimeoutError:
                        logger.warning(f"Task {task.task_id} timed out on attempt {attempt + 1}")
                        if attempt == task.max_retries:
                            break
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        
                    except Exception as e:
                        logger.error(f"Task {task.task_id} failed on attempt {attempt + 1}: {e}")
                        if attempt == task.max_retries:
                            break
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
                # All retries failed
                processing_time = time.time() - start_time
                progress_state.failed_tasks += 1
                progress_state.error_messages.append(f"Task {task.task_id} failed after {task.max_retries + 1} attempts")
                
                if progress_callback:
                    await progress_callback(progress_state)
                
                return ProcessingResult(
                    task_id=task.task_id,
                    success=False,
                    error=f"Failed after {task.max_retries + 1} attempts",
                    processing_time=processing_time
                )
        
        # Execute all tasks in the batch concurrently
        batch_coroutines = [process_single_task(task) for task in batch_tasks]
        results = await asyncio.gather(*batch_coroutines, return_exceptions=True)
        
        # Handle any exceptions from gather
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch task {batch_tasks[i].task_id} raised exception: {result}")
                processed_results.append(ProcessingResult(
                    task_id=batch_tasks[i].task_id,
                    success=False,
                    error=str(result)
                ))
            else:
                processed_results.append(result)
        
        # Store results for session tracking
        if progress_state.session_id in self.task_results:
            self.task_results[progress_state.session_id].extend(processed_results)
        
        return processed_results
    
    def _calculate_processing_stats(
        self,
        results: List[ProcessingResult],
        start_time: datetime
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive processing statistics.
        
        Args:
            results: List of processing results
            start_time: Processing start time
            
        Returns:
            Dictionary containing processing statistics
        """
        if not results:
            return {"total_tasks": 0, "success_rate": 0.0}
        
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        total_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        processing_times = [r.processing_time for r in results if r.processing_time > 0]
        
        stats = {
            "total_tasks": len(results),
            "successful_tasks": len(successful_results),
            "failed_tasks": len(failed_results),
            "success_rate": len(successful_results) / len(results),
            "total_processing_time": total_time,
            "average_task_time": sum(processing_times) / len(processing_times) if processing_times else 0.0,
            "min_task_time": min(processing_times) if processing_times else 0.0,
            "max_task_time": max(processing_times) if processing_times else 0.0,
            "tasks_per_second": len(results) / total_time if total_time > 0 else 0.0,
            "concurrent_efficiency": len(results) / (total_time * self.max_workers) if total_time > 0 else 0.0
        }
        
        # Update global performance metrics
        self.performance_metrics["average_task_time"] = (
            (self.performance_metrics["average_task_time"] * (self.performance_metrics["total_tasks_processed"] - len(results)) +
             sum(processing_times)) / self.performance_metrics["total_tasks_processed"]
        ) if self.performance_metrics["total_tasks_processed"] > 0 else 0.0
        
        self.performance_metrics["success_rate"] = (
            (self.performance_metrics["success_rate"] * (self.performance_metrics["total_tasks_processed"] - len(results)) +
             len(successful_results)) / self.performance_metrics["total_tasks_processed"]
        ) if self.performance_metrics["total_tasks_processed"] > 0 else 1.0
        
        return stats
    
    def get_progress_state(self, session_id: str) -> Optional[ProgressState]:
        """
        Get current progress state for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            ProgressState if session is active, None otherwise
        """
        return self.active_sessions.get(session_id)
    
    def get_session_results(self, session_id: str) -> List[ProcessingResult]:
        """
        Get processing results for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of processing results
        """
        return self.task_results.get(session_id, [])
    
    def cancel_session(self, session_id: str) -> bool:
        """
        Cancel processing for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was cancelled, False if not found
        """
        if session_id in self.active_sessions:
            progress_state = self.active_sessions[session_id]
            progress_state.current_stage = ProcessingStage.FAILED
            progress_state.error_messages.append("Processing cancelled by user")
            
            # Note: Individual tasks may still complete, but we mark session as cancelled
            logger.info(f"Session {session_id} marked as cancelled")
            return True
        
        return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics.
        
        Returns:
            Dictionary containing performance statistics
        """
        active_sessions_count = len(self.active_sessions)
        
        return {
            "global_metrics": self.performance_metrics.copy(),
            "configuration": {
                "max_workers": self.max_workers,
                "default_timeout": self.default_timeout
            },
            "current_state": {
                "active_sessions": active_sessions_count,
                "thread_pool_active": self.thread_pool is not None,
                "semaphore_capacity": self.semaphore._value if hasattr(self.semaphore, '_value') else None
            }
        }
    
    async def shutdown(self):
        """Shutdown the processor and cleanup resources."""
        logger.info("Shutting down async processor...")
        
        # Cancel all active sessions
        for session_id in list(self.active_sessions.keys()):
            self.cancel_session(session_id)
        
        # Shutdown thread pool if active
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
            self.thread_pool = None
        
        # Clear tracking data
        self.active_sessions.clear()
        self.task_results.clear()
        
        logger.info("Async processor shutdown complete")


# Global processor instance
async_processor = AsyncProcessor()


def get_async_processor() -> AsyncProcessor:
    """
    Get the global async processor instance.
    
    Returns:
        AsyncProcessor instance for dependency injection
    """
    return async_processor


async def create_progress_callback(
    callback_func: Optional[Callable] = None
) -> Callable[[ProgressState], None]:
    """
    Create a standardized progress callback function.
    
    Args:
        callback_func: Optional custom callback function
        
    Returns:
        Progress callback function
    """
    async def default_callback(progress_state: ProgressState):
        """Default progress callback that logs progress."""
        logger.info(
            f"Session {progress_state.session_id}: "
            f"{progress_state.current_stage.value} - "
            f"{progress_state.overall_progress:.1f}% complete "
            f"({progress_state.completed_tasks}/{progress_state.total_tasks})"
        )
    
    return callback_func or default_callback