
class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = ""
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'LEVIR':
            self.label_transform = "norm"
            self.root_dir = '/home/students/master/2023/xur/xr/LEVIR-CD/'
        elif data_name == 'WHU':
            self.label_transform = "norm"
            self.root_dir = '/home/students/master/2023/xur/xr/Dataset/WHU-CD/'
        elif data_name == 'CLCD':
            self.label_transform = "norm"
            self.root_dir = '/home/students/master/2023/xur/xr/Dataset/CLCD-256/'
        else:
            raise TypeError('%s has not defined' % data_name)
        return self


if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='LEVIR')
    print(data.data_name)
    print(data.root_dir)
    print(data.label_transform)

