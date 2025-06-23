import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List

from src.lib.context import Context
from src.lib.data import Data
from src.lib.pipeline import Pipeline
from src.steps import Fill, OCR, Parse, Populate



logger = logging.getLogger(__name__)

data = [
    Data(
        name="Adbulla",
        input_dir="input/adbulla",
        output_dir="output/adbulla",
    ),
    Data(
        name="Akshay",
        input_dir="input/akshay",
        output_dir="output/akshay",
    ),
    # Data(
    #     name="Amy",
    #     input_dir="input/amy",
    #     output_dir="output/amy",
    # ),
]


def prepare_data(data: List[Data]):
    """Prepare output directories for all data entries"""
    logger.info("Preparing data directories...")
    
    for data_entry in data:
        try:
            output_path = Path(data_entry.output_dir)
            input_path = Path(data_entry.input_dir)
            
            # Create output directory
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Check if input directory exists
            if not input_path.exists():
                logger.error(f"  Input directory does not exist: {input_path}")
                continue
                
        except Exception as e:
            logger.error(f"Error preparing data for {data_entry.name}: {str(e)}")


if __name__ == "__main__":
    
    log_path = Path("logs")
    log_path.mkdir(parents=True, exist_ok=True)
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'logs/pipeline_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    logger.info("="*80)
    logger.info("STARTING MEDICAL FORM PROCESSING PIPELINE")
    logger.info("="*80)
    
    try:
        # Prepare data directories
        prepare_data(data)
        
        # Create context
        logger.info("Creating pipeline context...")
        context = Context(data=data)
        
        # Create pipeline with steps
        logger.info("Initializing pipeline steps...")
        steps = [
            Parse(),
            OCR(),
            Populate(),
            Fill()
        ]
        
        for step in steps:
            logger.info(f"  Initialized step: {step.name} - {step.description}")
        
        pipeline = Pipeline(steps, context)
        logger.info(f"Pipeline created with {len(steps)} steps")
        
        # Run pipeline
        logger.info("Starting pipeline execution...")
        start_time = datetime.now()
        
        pipeline.run()
        
        end_time = datetime.now()
        execution_time = end_time - start_time
        
        logger.info("="*80)
        logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        logger.info(f"Total execution time: {execution_time}")
        logger.info("="*80)
        
    except Exception as e:
        logger.error("="*80)
        logger.error("PIPELINE EXECUTION FAILED")
        logger.error(f"Error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error("="*80)
        raise
