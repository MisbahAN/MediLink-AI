import logging
from datetime import datetime
from typing import List

from .context import Context
from .step import Step

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, steps: List[Step], context: Context):
        self.steps = steps
        self.context = context
        logger.info(f"Pipeline initialized with {len(steps)} steps")

    def run(self):
        """Execute all pipeline steps with comprehensive logging"""
        logger.info(f"Starting pipeline execution with {len(self.steps)} steps")
        
        total_start_time = datetime.now()
        successful_steps = 0
        failed_steps = 0
        
        for i, step in enumerate(self.steps, 1):
            step_start_time = datetime.now()
            logger.info(f"Step {i}/{len(self.steps)}: Starting {step.name}")
            
            try:
                # Pre-run phase
                logger.debug(f"  Executing pre_run for {step.name}")
                step.pre_run(self.context)
                
                # Main run phase
                logger.debug(f"  Executing main run for {step.name}")
                step.run(self.context)
                
                # Post-run phase
                logger.debug(f"  Executing post_run for {step.name}")
                step.post_run(self.context)
                
                step_end_time = datetime.now()
                step_duration = step_end_time - step_start_time
                successful_steps += 1
                
                logger.info(f"Step {i}/{len(self.steps)}: {step.name} completed successfully in {step_duration}")
                
            except Exception as e:
                step_end_time = datetime.now()
                step_duration = step_end_time - step_start_time
                failed_steps += 1
                
                logger.error(f"Step {i}/{len(self.steps)}: {step.name} FAILED after {step_duration}")
                logger.error(f"  Error: {str(e)}")
                logger.error(f"  Error type: {type(e).__name__}")
                
                # Decide whether to continue or stop
                if self._should_continue_on_error(step, e):
                    logger.warning(f"  Continuing pipeline execution despite {step.name} failure")
                else:
                    logger.error(f"  Stopping pipeline execution due to critical failure in {step.name}")
                    raise
        
        total_end_time = datetime.now()
        total_duration = total_end_time - total_start_time
        
        logger.info("="*60)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info(f"Total steps: {len(self.steps)}")
        logger.info(f"Successful steps: {successful_steps}")
        logger.info(f"Failed steps: {failed_steps}")
        logger.info(f"Total execution time: {total_duration}")
        logger.info("="*60)

    def _should_continue_on_error(self, step: Step, error: Exception) -> bool:
        """Determine if pipeline should continue after a step failure"""
        # For now, always stop on errors - can be customized per step type
        return False

    def add_step(self, step: Step):
        """Add a step to the pipeline"""
        self.steps.append(step)
        logger.info(f"Added step to pipeline: {step.name}")
