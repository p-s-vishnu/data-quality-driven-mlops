import util
from classification import classification

util.start_logging(cmd_out=True)
classification.main()
util.export_log_file_as_csv()