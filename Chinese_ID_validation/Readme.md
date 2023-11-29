## Translation Validation

### Annotator Details
- We collaborated with 2 Chinese Language Experts from ALA Language Center Company to validate the effectivity of the translations. Sufficient money was given to both the annotators (USD 0.04 per example per expert) for each example, which was sufficient considering the country of residence of the experts. 
- Both the annotators are from Asian countries with proficiency in English and Chinese Languages along with sufficient medical knowledge. They have previously contributed to similar research studies published at reputed Conferences and Journals.

### Annotation Process and Instructions
- We take a random sample of 400 examplese from CMID and 400 random examples from Dev set of KUAKE-QIC dataset for manual validaiton of Google API translaiton. The validation is done by two Chinese experts who are proficient in English, Chinese (HSK Level-3) and had the required medical domain knowledge. 

- Annotators were instructed to label each translation in one of three categories (mentioned in "validation" column in the sheets):
    ``` 
    1  : Translation is accurate
    0  : Translation closely resembles original meaning
    -1 : Translation is inaccurate
    ```
- We aggregate the 1 and 0 class to represent correct translation and -1 is taken as incorrect translation.

- We report inter-annotator agreement Cohen Kappa on initial annotation. All the conflicts were later resolved with discussions and final validation result sheets are shared on this page.
