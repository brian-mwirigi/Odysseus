import sys

with open('plotting.ipynb', 'r', encoding='utf-8') as f:
    content = f.read()

replacements = [
    (
'''<<<<<<< HEAD
    "plt.savefig('Plot_2_Shock_vs_Status.png',dpi=150, bbox_inches='tight')\\n",
=======
    "# plt.savefig('Plot_2_Shock_vs_Status.png', dpi=150, bbox_inches='tight')\\n",
>>>>>>> ecdadff1a538fbc6ac85b413dc978312bac08d68''',
'''    "plt.savefig('Plot_2_Shock_vs_Status.png',dpi=150, bbox_inches='tight')\\n",'''
    ),
    (
'''<<<<<<< HEAD
      "Difference (None correct vs All correct \u2014 Worsened %):\\n",
      "  -10.6 percentage points\\n"
=======
      "Difference (None correct vs All correct — Worsened %):\\n"
>>>>>>> ecdadff1a538fbc6ac85b413dc978312bac08d68''',
'''      "Difference (None correct vs All correct \u2014 Worsened %):\\n",
      "  -10.6 percentage points\\n"'''
    ),
    (
'''<<<<<<< HEAD
    "print(f\\"\\\\nDifference (None correct vs All correct \u2014 Worsened %):\\")\\n",
    "diff = fl_ct.loc['None correct','Worsened'] - fl_ct.loc['All correct','Worsened']\\n",
    "print(f\\"  {diff:.1f} percentage points\\")"
=======
    "print(f\\"\\\\nDifference (None correct vs All correct — Worsened %):\\")\\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2ed9cbc9",
   "metadata": {},
   "source": [
    "##Insights\\n",
    "\\n",
    "Financial literacy shows a gradual but meaningful relationship with financial outcomes. Respondents who answered \\"All correct\\" on the literacy assessment have a higher share of \\"Improved\\" status compared to those with \\"None correct.\\" while those with zero correct answers are excessively represented in the \\"Worsened\\" group. This suggests that financial literacy is an important factor as people who understand financial concepts are better equipped to make decisions that improve their situation."
>>>>>>> ecdadff1a538fbc6ac85b413dc978312bac08d68''',
'''    "print(f\\"\\\\nDifference (None correct vs All correct \u2014 Worsened %):\\")\\n",
    "diff = fl_ct.loc['None correct','Worsened'] - fl_ct.loc['All correct','Worsened']\\n",
    "print(f\\"  {diff:.1f} percentage points\\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2ed9cbc9",
   "metadata": {},
   "source": [
    "##Insights\\n",
    "\\n",
    "Financial literacy shows a gradual but meaningful relationship with financial outcomes. Respondents who answered \\"All correct\\" on the literacy assessment have a higher share of \\"Improved\\" status compared to those with \\"None correct.\\" while those with zero correct answers are excessively represented in the \\"Worsened\\" group. This suggests that financial literacy is an important factor as people who understand financial concepts are better equipped to make decisions that improve their situation."'''
    )
]

for t, r in replacements:
    if t in content:
        content = content.replace(t, r)
        print('Replaced a block!')
    else:
        print('Could not find block starting with:', t[:40])

with open('plotting.ipynb', 'w', encoding='utf-8') as f:
    f.write(content)
