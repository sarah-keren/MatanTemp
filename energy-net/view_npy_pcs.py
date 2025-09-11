
import numpy as np
import matplotlib.pyplot as plt


for i in range(1,7):

    pcs_actions = np.load(f"vladimir/b{i}_actions.npy")

    plt.scatter(range(48), pcs_actions[:48])  # points only
    plt.xlabel("Step"); plt.ylabel("PCS Action")
    plt.title(f"PCS Actions for agent {i} (48 steps)")
    plt.grid(True)

    plt.savefig(f"vladimir/b{i}_actions_plot.png", dpi=300, bbox_inches="tight")  # save first
    plt.show()                                                         # then display
    plt.close()                                                      # optional: close figure












#pcs_actions = np.load("vladimir/b6_actions.npy")
#print(pcs_actions[:48])

##plt.plot(pcs_actions[:48])   # first 200 actions
#plt.scatter(range(len(pcs_actions[:48])), pcs_actions[:48])  # points only

#plt.xlabel("Step")
#plt.ylabel("PCS Action")
#plt.title("PCS Actions (first 48 steps)")
#plt.grid(True)
#plt.show()
#plt.savefig("b6_actions_plot.png", dpi=300, bbox_inches="tight")  # save to file
#plt.close()



