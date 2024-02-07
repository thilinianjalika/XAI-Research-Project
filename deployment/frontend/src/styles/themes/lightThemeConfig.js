import { deepMerge } from "@/functions/util";
import base from "./base";

const overWrite = {
    palette: {
        mode: "light",
    },
};

export default deepMerge(base, overWrite);
