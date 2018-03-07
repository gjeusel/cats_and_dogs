from pathlib import Path
import pandas as pd


# (1 = dog, 0 = cat)

df = pd.DataFrame(columns={'label'})
counter = 0
for f in (Path(__file__).parent / 'train').iterdir():
    print('Processing {}...'.format(f))
    if f.name[:4] == 'dog.':
        df = df.append({'label': 1}, ignore_index=True)
    elif f.name[:4] == 'cat.':
        df = df.append({'label': 0}, ignore_index=True)
    else:
        raise ValueError()
    f.rename(f.parent / '{}.jpg'.format(counter))
    counter += 1

df.index.name = 'id'
df.to_csv(Path(__file__).parent / 'train.csv')
