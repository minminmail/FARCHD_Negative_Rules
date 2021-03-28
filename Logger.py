
import logging

class Logger:
    
    @staticmethod
    def set_logger():


        logging.basicConfig(filename = "help.log",format='%(asctime)s %(message)s', 
					filemode='w'  )
        logger=logging.getLogger()
        """  
        logger.propagate = False
        logger.setLevel(logging.DEBUG)
        """


        return logger


