import {
    MODEL_NAME_KNN,
    MODEL_NAME_LR,
    MODEL_NAME_RF,
    MODEL_NAME_SVM,
} from "@/constants";
import commonStyles from "@/styles/commonStyles";
import { Autocomplete, Box, TextField } from "@mui/material";

const ModelSelection = ({ setModel }) => {
    const models = [
        { label: "K Nearest Neighbour Model", value: MODEL_NAME_KNN },
        { label: "Linear Regression Model", value: MODEL_NAME_LR },
        { label: "Random Forest Model", value: MODEL_NAME_RF },
        { label: "Support Vector Machine Model", value: MODEL_NAME_SVM },
    ];
    return (
        <Box sx={{ width: "300px" }}>
            <Autocomplete
                disablePortal
                onChange={(e, obj) => (obj ? setModel(obj.value) : setModel())}
                options={models}
                sx={commonStyles.autocomplete}
                isOptionEqualToValue={(obj) => obj.label}
                size="small"
                renderInput={(params) => (
                    <TextField {...params} label="Models" />
                )}
            />
        </Box>
    );
};

export default ModelSelection;
