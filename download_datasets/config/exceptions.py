class ConfigError(KeyError):
    def __init__(self, missing_attribute, config_name):
        super().__init__(f'Bad configuration format: missing "{missing_attribute}" attribute in "{config_name}"')


class DownloaderConfigError(KeyError):
    def __init__(self, downloader_attribute, config_name):
        super().__init__(f'Bad configuration format: type "{downloader_attribute}" in "{config_name}" does not exist')


class PostprocessConfigError(KeyError):
    def __init__(self, postprocessor_attribute, config_name):
        super().__init__(f'Bad configuration format: processor "{postprocessor_attribute}" in "{config_name}" does not exist')
