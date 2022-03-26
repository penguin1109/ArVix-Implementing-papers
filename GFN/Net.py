class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.deblurMoudle      = self._make_net(_DeblurringMoudle)
        self.srMoudle          = self._make_net(_SRMoudle)
        self.geteMoudle        = self._make_net(_GateMoudle)
        self.reconstructMoudle = self._make_net(_ReconstructMoudle)

    def forward(self, x, gated, isTest):
        if isTest == True:
            origin_size = x.size()
            input_size  = (math.ceil(origin_size[2]/4)*4, math.ceil(origin_size[3]/4)*4)
            out_size    = (origin_size[2]*4, origin_size[3]*4)
            x           = nn.functional.upsample(x, size=input_size, mode='bilinear')

        deblur_feature, deblur_out = self.deblurMoudle(x)
        sr_feature = self.srMoudle(x)
        if gated == True:
            scoremap = self.geteMoudle(torch.cat((deblur_feature, x, sr_feature), 1))
        else:
            scoremap = torch.cuda.FloatTensor().resize_(sr_feature.shape).zero_()+1
        repair_feature = torch.mul(scoremap, deblur_feature)
        fusion_feature = torch.add(sr_feature, repair_feature)
        recon_out = self.reconstructMoudle(fusion_feature)

        if isTest == True:
            recon_out = nn.functional.upsample(recon_out, size=out_size, mode='bilinear')

        return deblur_out, recon_out

    def _make_net(self, net):
        nets = []
        nets.append(net())
        return nn.Sequential(*nets)
