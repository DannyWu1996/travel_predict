class Options():
    def __init__(
            self,
            aux_path,
            data_name,
            cuda="0",
            method="ggnn",
            validation=False,
            train_batch_size=32,
            predict_batch_size=32,
            max_epoch=10,
            top_k=5,
            lr=1e-4,
            hiddenSize=100,
            l2=1e-5,
            step=1,
            lr_dc=0.1,
            lr_dc_step=3,
            nonhybrid = False,
            is_restore = False,
    ):
        self.cuda = cuda
        self.aux_path = aux_path
        self.data_name = data_name
        self.method = method
        self.validation = validation
        self.train_batch_size = train_batch_size
        self.predict_batch_size = predict_batch_size
        self.max_epoch = max_epoch
        self.top_k = top_k
        self.lr = lr
        self.hiddenSize = hiddenSize
        self.l2 = l2
        self.step = step
        self.nonhybrid = nonhybrid
        self.lr_dc = lr_dc
        self.lr_dc_step = lr_dc_step
        self.is_restore = is_restore