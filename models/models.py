
def create_model(opt):
    print(opt.model)

    # our anatomy-preserving adaptation segmentation
    if opt.model == 'apada2seg_model_train':
        assert(opt.dataset_mode == 'apada2seg_train')
        from .apada2seg_model import APADA2SEGModel
        model = APADA2SEGModel()
    elif opt.model == 'apada2seg_model_test':
        assert(opt.dataset_mode == 'apada2seg_test')
        from .apada2seg_model import APADA2SEGModel_TEST
        model = APADA2SEGModel_TEST()

    else:
        raise ValueError("Model [%s] not recognized." % opt.model)

    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
