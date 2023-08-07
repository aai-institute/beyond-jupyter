# Step 2: Dataset Representation

In this step, we introduce a representation for the data set:
 * **We explicitly represent the parameters that determine the data** in the attributes  
   of a newly introduced class `Dataset`, making
   our choices explicit: 
   * We don't really agree with the notion of dropping songs with zero popularity,
     as it basically amounts to cheating. In reality, many songs have zero popularity
     and our model should be able to handle it and ideally be able to predict it, too.
     Therefore, we have made this an option, which is disabled by default.
   * We furthermore added the threshold for a song being considered as popular
     as an explicit parameter. We might want to change it down the line, noting 
     that any hard threshold is somewhat arbitrary in the end. Perhaps we'll consider treating
     this as a regression problem instead and avoid the problem entirely.
   * We also allow to sample the data to speed up development. In many stages
     of development, we don't care about learning a very good model that considers
     as much data as possible. So for experimentation, where we want to check
     only if our code works and perhaps get a rough estimate of how well it works,
     we'll be using a smaller sample of the data.
   We will only be using a single data set in this project, but we will still experiment with 
   different variations thereof. Therefore, we need to make the respective parameters 
   explicit. The parameters collectively determine the concrete manifestation of the data
   and recording them will therefore be akin to data versioning.
 * **We shall prefer to keep the data
   untouched and make models decide what to do with it**. 
   We don't need to drop or modify any data columns from the get-go. Models can just 
   choose to use only a subset of the data. Different models shall
   be able to do different things entirely.
 * **We avoid the use of constant literals in our code and use named constants instead.**
     * Referring to data columns via string literals is prone to errors; typos happen.
     * We get no assistance from our IDE when we're in doubt as to how a column is named. 
       By adding constants to our code with a well-defined identifier scheme (all identifiers start with `COL_`), we
       get optimal assistance from our IDE: By typing `COL_` and asking for an auto-completion,
       we get a list of possible options. 
     * If the column names should ever change, we have a single place in which we would need to 
       update them (though this won't be an issue in this toy project, of course).
     * In addition to the original columns, we introduce constants for the columns we add (such as 
       the class column, with dedicated prefix `COL_GEN`) and introduce semantic groups (with the dedicated prefix `COLS_`) that could
       come in handy later on. 
 * We simplified the generation of the class column using the `apply` function.
 * We have made explicit the set of features that is actually being used by the models (in variable `cols_used_by_models`).
   Previously, it was simply the set of numerical columns that remained after some drops and projections.
   And we can already see that applying StandardScaler to columns that are actually categorical (such as the key and the mode),
   is, at the very least, questionable. We shall do something about this in a subsequent step.

# Principles Addressed in this Step

* Find the right abstractions
* Expose parametrisation
* (Version data)