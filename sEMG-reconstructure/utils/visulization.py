import matplotlib.pyplot as plt

def reconvisualization(orgframe, compframe, recomframe, fig_path, epoch):
    orgframe = orgframe.cpu().detach().numpy()
    compframe = compframe.cpu().detach().numpy()
    recomframe = recomframe.cpu().detach().numpy()
    plt.figure(figsize=(12, 16))

    plt.subplot(311)
    plt.plot(orgframe)
    plt.legend(["original singal"])
    plt.grid()
    plt.subplot(312)
    plt.plot(compframe)
    plt.legend(["compressed singal"])
    plt.grid()
    plt.subplot(313)
    plt.plot(recomframe)
    plt.legend(["rebuilt singal"])
    plt.grid()
    plt.savefig(fig_path+str(epoch)+'.eps', format='eps', dpi=300)
    # plt.savefig("filename.svg",format="svg",transparent=True)
    #plt.show()


