import {
    MODEL_NAME_KNN,
    MODEL_NAME_LR,
    MODEL_NAME_RF,
    MODEL_NAME_SVM,
} from "@/constants";
import { Table } from "@mui/material";
import SVMKNN from "./SVMKNN";
import RFLR from "./RFLR";

const ConfigTable = ({ configurations, setConfigurations, model }) => {
    const handleDelete = (i) => {
        const newConfigs = [...configurations];
        newConfigs.splice(i, 1);
        setConfigurations(newConfigs);
    };
    if (model === MODEL_NAME_KNN || model === MODEL_NAME_SVM) {
        return (
            <SVMKNN
                configurations={configurations}
                handleDelete={handleDelete}
            />
        );
    } else if (model === MODEL_NAME_RF || model === MODEL_NAME_LR) {
        return (
            <RFLR configurations={configurations} handleDelete={handleDelete} />
        );
    } else {
        return <Table></Table>;
    }
};

export default ConfigTable;
