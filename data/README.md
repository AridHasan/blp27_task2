# Data: BLP-2027 Task 2 (Multilingual Cultural QA)

The data comes from [MultiNativQA](https://huggingface.co/datasets/QCRI/MultiNativQA) ([Hasan et al., 2025](https://aclanthology.org/2025.findings-acl.770/)), a multilingual, native and culturally aligned QA resource built from real queries by native speakers and manually annotated answers. Questions cover 18 topics: Animals, Business, Clothing, Education, Events, Food & Drinks, General, Geography, Immigration, Language, Literature, Names & Persons, Plants, Religion, Sports & Games, Tradition, Travel and Weather.


| Language | MultiNativQA configs | Train | Dev | Dev-Test |
|---|---|---:|---:|---:|
| Bangla (`bn`) | `bangla_bd` (Dhaka), `bangla_in` (Kolkata) | 13,909 | 1,883 | 3,667 |
| English (`en`) | `english_bd` (Dhaka), `english_qa` (Doha) | 12,973 | 1,820 | 3,435 |
| Assamese (`as`) | `assamese_in` (Assam) | 1,131 | 157 | 545 |
| Hindi (`hi`) | `hindi_in` (Delhi) | 9,288 | 1,286 | 2,745 |
| **Total** | | **37,301** | **5,146** | **10,392** |

- **train / dev**: the MultiNativQA train and dev splits.
- **dev_test**: the MultiNativQA test split, used for the development-phase leaderboard. It must **not** be used for training.
- **test** (evaluation phase): a new blind test set per language, released without answers.

## Format

JSON Lines, UTF-8, one object per line:

| Field | Description |
|---|---|
| `id` | Unique id: `<lang>_<region>_<MultiNativQA data_id>` |
| `lang` | `bn`, `en`, `as` or `hi` |
| `region` | `bd` (Bangladesh), `in` (India), `qa` (Qatar) |
| `category` | Topic of the question |
| `question` | The question |
| `answer` | Reference answer (absent in the blind test set) |

## License

CC BY-NC-SA 4.0, following MultiNativQA.
